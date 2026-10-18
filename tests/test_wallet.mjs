// No wallet connection, signatures or network requests: validate the unsigned payment contents.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { webcrypto } from 'node:crypto';
const context = vm.createContext({console, TextEncoder, TextDecoder, Uint8Array, ArrayBuffer, Buffer, crypto: webcrypto, setTimeout, clearTimeout});
vm.runInContext(readFileSync(new URL('../static/vendor/solana-web3-1.98.4.js', import.meta.url), 'utf8'), context);
vm.runInContext(`
const sdk = solanaWeb3;
const payer = new sdk.PublicKey(new Uint8Array(32).fill(2));
const recipient = new sdk.PublicKey(new Uint8Array(32).fill(3));
const reference = new sdk.PublicKey(new Uint8Array(32).fill(4));
const transfer = sdk.SystemProgram.transfer({fromPubkey: payer, toPubkey: recipient, lamports: 10000000});
transfer.keys.push({pubkey: reference, isSigner: false, isWritable: false});
const tx = new sdk.Transaction({feePayer: payer, recentBlockhash: new sdk.PublicKey(new Uint8Array(32).fill(5)).toString()}).add(transfer);
tx.add(new sdk.TransactionInstruction({keys: [], programId: new sdk.PublicKey('MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr'), data: new TextEncoder().encode('invoice-demo')}));
const serialized = tx.serializeMessage();
const decoded = sdk.Message.from(serialized);
globalThis.paymentCheck = {
  signers: decoded.header.numRequiredSignatures,
  amount: sdk.SystemInstruction.decodeTransfer(transfer).lamports.toString(),
  destination: sdk.SystemInstruction.decodeTransfer(transfer).toPubkey.toString(),
  expectedDestination: recipient.toString(),
  referenceIncluded: decoded.accountKeys.some(k => k.equals(reference)),
  programs: decoded.instructions.map(i => decoded.accountKeys[i.programIdIndex].toString()),
};
`, context);
assert.equal(context.paymentCheck.signers, 1);
assert.equal(context.paymentCheck.amount, '10000000');
assert.equal(context.paymentCheck.destination, context.paymentCheck.expectedDestination);
assert.equal(context.paymentCheck.referenceIncluded, true);
assert.deepEqual(Array.from(context.paymentCheck.programs), ['11111111111111111111111111111111', 'MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr']);
console.log('Unsigned SOL transfer, memo, reference and signer verified. No funds sent.');
