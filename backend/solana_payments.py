"""Non-custodial SOL payment intents and finalized receipt verification. Never signs."""
from decimal import Decimal, InvalidOperation
import json
import os
import re
import secrets
import time
import urllib.parse
import urllib.request
from .http_cache import _context
from .conferences import page

ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
SYSTEM_PROGRAM = '11111111111111111111111111111111'
GENESIS = {'devnet': 'EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG', 'mainnet-beta': '5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d'}
NETWORKS = {'devnet': 'https://api.devnet.solana.com', 'mainnet-beta': 'https://api.mainnet-beta.solana.com'}


def b58encode(raw):
    number = int.from_bytes(raw, 'big')
    result = ''
    while number:
        number, index = divmod(number, 58)
        result = ALPHABET[index] + result
    return '1' * (len(raw) - len(raw.lstrip(b'\0'))) + result


def b58decode(value):
    if not isinstance(value, str) or not value or len(value) > 100 or any(c not in ALPHABET for c in value):
        raise ValueError('Invalid base58 value.')
    number = 0
    for char in value:
        number = number * 58 + ALPHABET.index(char)
    raw = number.to_bytes((number.bit_length()+7)//8, 'big')
    return b'\0' * (len(value)-len(value.lstrip('1'))) + raw


def address(value):
    if len(b58decode(value)) != 32 or value == SYSTEM_PROGRAM:
        raise ValueError('Use a valid non-system Solana recipient address.')
    return value


def network():
    selected = os.environ.get('SOLANA_NETWORK', 'mainnet-beta')
    if selected not in NETWORKS:
        raise ValueError('SOLANA_NETWORK must be devnet or mainnet-beta.')
    return selected


def rpc(method, params):
    # Endpoint is operator-configured, never taken from a search result or model response.
    endpoint = os.environ.get('SOLANA_RPC_URL', NETWORKS[network()])
    body = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}).encode()
    request = urllib.request.Request(endpoint, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=15, context=_context() if endpoint.startswith('https') else None) as response:
        payload = json.load(response)
    if 'error' in payload:
        raise ValueError('Solana RPC could not complete the request.')
    return payload.get('result')


def check_rpc_network():
    if rpc('getGenesisHash', []) != GENESIS[network()]:
        raise ValueError('RPC network does not match the displayed payment network. No payment is allowed.')


def latest_blockhash():
    check_rpc_network()
    return rpc('getLatestBlockhash', [{'commitment': 'finalized'}])


def parse_invoice(value):
    parsed = urllib.parse.urlparse(str(value).strip())
    if parsed.scheme != 'solana' or parsed.netloc or parsed.fragment:
        raise ValueError('Use an organizer-issued Solana Pay transfer URL, not a transaction-request URL.')
    recipient = address(parsed.path)
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    if any(len(v) != 1 for v in query.values()) or set(query) - {'amount', 'label', 'message', 'memo', 'reference'}:
        raise ValueError('Only native SOL transfer invoices are supported; unsupported tokens or invoice parameters were supplied.')
    try:
        amount_text = query['amount'][0]
        if not re.fullmatch(r'\d{1,10}(?:\.\d{1,9})?', amount_text):
            raise ValueError('Invoice amount must be a decimal SOL amount with at most nine fractional digits.')
        amount = Decimal(amount_text)
        if not amount.is_finite() or amount <= 0 or amount * 10**9 != (amount * 10**9).to_integral_value():
            raise ValueError('SOL amounts must be positive with at most nine decimal places.')
        lamports = int(amount * 10**9)
        if lamports > 2**53 - 1:
            raise ValueError('Amount exceeds the supported payment range.')
    except (KeyError, InvalidOperation) as exc:
        raise ValueError('The invoice needs a fixed SOL amount.') from exc
    if 'reference' in query:
        address(query['reference'][0])
    memo = query.get('memo', [''])[0]
    if len(memo.encode()) > 160:
        raise ValueError('Invoice memo is too long.')
    return {'recipient': recipient, 'amount': format(amount, 'f'), 'lamports': lamports, 'memo': memo,
            'merchant_reference': query.get('reference', [None])[0]}


def invoice_evidence(invoice_url, source_url):
    expected = parse_invoice(invoice_url)
    source = page(source_url)
    if not any(parse_invoice_safe(link) == expected for link in source['payment_links']):
        raise ValueError('The source page does not advertise this exact SOL invoice. Use the organizer’s normal checkout if it does not accept Solana Pay.')
    return expected, {'url': source['url'], 'checked_at': source['fetched_at']}


def parse_invoice_safe(value):
    try:
        return parse_invoice(value)
    except ValueError:
        return None


def create_intent(invoice, evidence, conference, manuscript, fee_kind, payer=None):
    reference = b58encode(secrets.token_bytes(32))
    now = time.time()
    intent = {**invoice, 'id': secrets.token_hex(12), 'reference': reference, 'network': network(), 'status': 'awaiting_wallet',
              'created_at': now, 'expires_at': now + 900, 'conference_id': conference['id'], 'conference_name': conference['name'],
              'conference_url': conference['url'], 'manuscript_id': manuscript['id'], 'manuscript_hash': manuscript['sha256'],
              'fee_kind': fee_kind, 'evidence': evidence, 'signature': None, 'payer': payer}
    params = [('amount', invoice['amount']), ('reference', reference), ('label', conference['name']), ('message', f'Conference {fee_kind} fee')]
    if invoice.get('merchant_reference'):
        params.append(('reference', invoice['merchant_reference']))
    if invoice['memo']:
        params.append(('memo', invoice['memo']))
    intent['payment_url'] = 'solana:' + invoice['recipient'] + '?' + urllib.parse.urlencode(params)
    return intent


def verify_receipt(intent, signature):
    if len(b58decode(signature)) != 64:
        raise ValueError('Invalid transaction signature.')
    if intent['network'] != network():
        raise ValueError('Payment belongs to a different configured network.')
    check_rpc_network()
    tx = rpc('getTransaction', [signature, {'encoding': 'jsonParsed', 'commitment': 'finalized', 'maxSupportedTransactionVersion': 0}])
    if tx is None:
        return None
    if not tx.get('meta') or tx['meta'].get('err') is not None:
        raise ValueError('The transaction failed.')
    if not tx.get('blockTime') or not intent['created_at'] - 60 <= tx['blockTime'] <= intent['expires_at'] + 300:
        raise ValueError('Transaction is outside this invoice’s payment window.')
    message = tx['transaction']['message']
    keys = message['accountKeys']
    key_names = {k['pubkey'] for k in keys}
    required = {intent['reference']}
    if intent.get('merchant_reference'):
        required.add(intent['merchant_reference'])
    if not required <= key_names:
        raise ValueError('Transaction does not contain the invoice reference.')
    instructions = message['instructions']
    allowed_programs = {SYSTEM_PROGRAM, 'ComputeBudget111111111111111111111111111111', 'MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr'}
    if any(i.get('programId') not in allowed_programs for i in instructions):
        raise ValueError('Receipt contains an unexpected program.')
    transfers = [i for i in instructions if i.get('programId') == SYSTEM_PROGRAM and (i.get('parsed') or {}).get('type') == 'transfer']
    if any(i.get('programId') == SYSTEM_PROGRAM and (i.get('parsed') or {}).get('type') != 'transfer' for i in instructions):
        raise ValueError('Receipt contains an unexpected system operation.')
    if len(transfers) != 1:
        raise ValueError('Expected exactly one SOL transfer.')
    info = transfers[0]['parsed']['info']
    if info.get('destination') != intent['recipient'] or info.get('lamports') != intent['lamports']:
        raise ValueError('Amount or recipient differs from the approved invoice.')
    if intent.get('payer') and info.get('source') != intent['payer']:
        raise ValueError('Payment came from a different wallet.')
    if not any(k['pubkey'] == info.get('source') and k.get('signer') for k in keys):
        raise ValueError('Transfer source is not a transaction signer.')
    if intent['memo'] and not any(i.get('program') == 'spl-memo' and i.get('parsed') == intent['memo'] for i in instructions):
        raise ValueError('Organizer invoice memo is missing.')
    return {'signature': signature, 'payer': info['source'], 'slot': tx['slot'], 'block_time': tx['blockTime'], 'verified_at': time.time(),
            'explorer': f'https://explorer.solana.com/tx/{signature}' + ('?cluster=devnet' if intent['network'] == 'devnet' else '')}
