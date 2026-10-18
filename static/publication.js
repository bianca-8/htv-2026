/* Conference discovery, manuscript review, and non-custodial registration payments. */
const pendingWallets = new Set();

function sourceUrl(value) {
  try { const u = new URL(value); return u.protocol === "https:" ? u.href : "#"; } catch { return "#"; }
}

function publicationHtml() {
  const pub = state.research.publication || {conferences: [], manuscripts: [], payments: [], search_status: "idle", network: "mainnet-beta"};
  const selected = pub.conferences.find(c => c.id === pub.selected_conference_id);
  const manuscript = pub.manuscripts.find(m => m.id === pub.active_manuscript_id);
  const accepted = state.research.runs.filter(r => r.status === "completed" && r.review === "accepted");
  const devnet = pub.network === "devnet";
  return `<section class="section publication"><p class="num">05 — Publication</p><h2>Find a home for the paper</h2>
    <p class="origin">Match your ideas to conference calls, review a manuscript, then handle registration after acceptance.</p>
    <button type="button" data-find-conferences ${pub.search_status === "searching" ? "disabled" : ""}>${pub.search_status === "searching" ? "Searching conference calls…" : "Find matching conferences"}</button>
    ${pub.search_error ? `<p class="error">${esc(pub.search_error)}</p>` : ""}
    ${pub.search_status === "outdated" ? `<p class="origin">Your ideas changed during the search. Search again for current matches.</p>` : ""}
    <details data-keep="conference-source"><summary>Search a specific call for papers</summary>
      <form data-publication-form="conference-source"><label>Official conference or call-for-papers URL<input type="url" name="source_url" placeholder="https://conference.org/call-for-papers" required></label><button type="submit">Read this conference call</button></form></details>
    ${pub.searches?.length ? `<p class="origin">Last search: ${esc(pub.searches.at(-1).query)}${pub.searches.at(-1).source_mode === "official_source_sweep" ? " · public search unavailable; read official conference sources" : ""}<br>${esc(pub.searches.at(-1).notice)}</p>` : ""}
    ${pub.search_status === "done" && !pub.conferences.length ? `<p class="origin">No source-backed conference match was returned. Try an official call-for-papers URL above.</p>` : ""}
    ${pub.conferences.map(c => `<article class="research-card conference-card"><p class="kicker">${selected?.id === c.id ? "Selected conference" : "Candidate · needs human review"}</p>
      <h3>${esc(c.name)}</h3>${c.deadline_status === "past_date_quoted" ? `<p class="error">The quoted deadline date has passed. Check the next edition or another submission track.</p>` : `<p class="origin">${c.deadline_status === "future_date_quoted" ? "A future deadline date is quoted; confirm the track and timezone." : "Confirm the current submission deadline and timezone on the organizer’s site."}</p>`}<p>${esc(c.fit)}</p><a href="${esc(sourceUrl(c.url))}" target="_blank" rel="noopener">Read the source and verify the organizer</a>
      <dl>${c.details.map(d => `<dt>${esc(d.kind)}</dt><dd>${esc(d.quote)}</dd>`).join("")}</dl>
      <p class="origin">Checked ${esc(c.checked_at)}. Dates and fees are source quotations; confirm the edition and timezone. A topic match does not predict acceptance.</p>
      <form data-publication-form="select-conference-${esc(c.id)}"><input type="hidden" name="conference_id" value="${esc(c.id)}">
        <label class="check-label"><input type="checkbox" name="organizer_confirmed" required>I checked the official organizer, current edition, deadline and paper requirements.</label>
        <button type="submit">Choose this conference</button></form></article>`).join("")}
    <h2 class="publication-step">Create the manuscript</h2>
    <p class="origin">Drafts include only completed results you accepted. The agent does not submit the paper or invent an acceptance.</p>
    <details data-keep="create-manuscript"><summary>${manuscript ? "Start another manuscript draft" : "Draft a paper from reviewed results"}</summary>
      <form data-publication-form="create-manuscript"><label>Paper title<input type="text" name="title" maxlength="300" required></label>
        <label>Author names<input type="text" name="authors" maxlength="1000" required></label>
        <p class="origin">${accepted.length} accepted experiment${accepted.length === 1 ? "" : "s"} available. Drafts still require your writing, verification, and conference formatting.</p>
        <button type="submit" ${accepted.length ? "" : "disabled"}>Create manuscript draft</button></form></details>
    ${manuscript ? manuscriptHtml(manuscript) : ""}
    <h2 class="publication-step">Registration after acceptance</h2>
    <p class="origin">${devnet ? "Devnet mode: test SOL only. A confirmed test transfer does not purchase real conference registration." : "Solana mainnet: real SOL. Only an organizer-issued Solana Pay invoice can be used."} The amount and recipient are shown before your wallet asks for approval.</p>
    ${selected && manuscript?.status === "reviewed" ? `<form data-publication-form="record-acceptance"><label>Organizer’s acceptance or submission reference<input type="text" name="reference" maxlength="1000" required></label>
      <label class="check-label"><input type="checkbox" name="confirmed" required>${devnet ? "I am recording an acceptance for this devnet demonstration." : "The selected conference has accepted this manuscript."}</label><button type="submit">Record acceptance</button></form>` : `<p class="origin">Choose a conference, complete the manuscript, and mark it reviewed to continue.</p>`}
    ${selected?.acceptance && manuscript?.status === "reviewed" ? `<p class="origin">Acceptance recorded by you: ${esc(selected.acceptance.reference)}</p>
      <details data-keep="registration-invoice"><summary>Prepare a ${devnet ? "devnet test" : "conference"} registration payment</summary>
      <form data-publication-form="registration-invoice">${devnet ? `<label>Demo organizer’s receiving wallet address<input type="text" name="recipient" minlength="32" maxlength="44" required></label><label>Registration amount in test SOL<input type="text" name="amount" inputmode="decimal" placeholder="0.01" required></label><p class="origin">This is the test recipient. Your paying wallet connects in the next step. Do not enter private keys or seed phrases.</p>` : `<label>Organizer-issued Solana Pay invoice URL<input type="text" name="invoice_url" placeholder="solana:RECIPIENT?amount=0.01" required></label><label>Official payment page containing this invoice<input type="url" name="source_url" required></label>`}
        <label class="check-label"><input type="checkbox" name="${devnet ? "demo_confirmed" : "organizer_payment_confirmed"}" required>${devnet ? "This uses test SOL and does not register me for a real conference." : "I verified this invoice and recipient with the official conference organizer."}</label>
        <button type="submit">Prepare payment for review</button></form></details>` : ""}
    ${(pub.payments || []).map(p => paymentHtml(p)).join("")}
  </section>`;
}

function manuscriptHtml(m) {
  return `<article class="research-card"><p class="kicker">Manuscript · version ${m.version} · ${esc(m.status)}</p><h3>${esc(m.title)}</h3>
    <p>${esc(m.authors)}</p><a href="/api/investigations/${state.id}/manuscripts/${m.id}.md">Download manuscript (Markdown)</a>
    <details data-keep="manuscript-${m.id}"><summary>Read and edit the manuscript</summary>
      <form data-publication-form="save-manuscript"><input type="hidden" name="manuscript_id" value="${m.id}"><input type="hidden" name="version" value="${m.version}">
        <label>Manuscript content<textarea name="content" rows="22" maxlength="100000" required>${esc(m.content)}</textarea></label>
        <button type="submit">Save a new version</button></form></details>
    <form data-publication-form="review-manuscript"><input type="hidden" name="manuscript_id" value="${m.id}"><input type="hidden" name="version" value="${m.version}">
      <label class="check-label"><input type="checkbox" name="confirmed" required>I reviewed the current text, results, citations, authorship and limitations, and completed the draft sections.</label>
      <button type="submit">Mark manuscript reviewed</button></form></article>`;
}

function paymentHtml(p) {
  const networkMismatch = p.network !== (state.research.publication?.network || "mainnet-beta");
  const busy = pendingWallets.has(p.id) || networkMismatch;
  const devnet = p.network === "devnet";
  const signature = p.signature || localStorage.getItem(`marginalia-payment-${p.id}`) || "";
  return `<article class="research-card payment-card"><p class="kicker">${esc(p.status.replaceAll("_", " "))} · ${devnet ? "DEVNET TEST" : "MAINNET · REAL SOL"}</p>
    <h3>${esc(p.conference_name)} registration</h3><p class="payment-amount">${esc(p.amount)} ${devnet ? "test SOL" : "SOL"} <small>+ wallet-displayed network fee</small></p>
    <dl><dt>Recipient — verify before signing</dt><dd><code>${esc(p.recipient)}</code></dd><dt>Network</dt><dd>${esc(p.network)}</dd><dt>Manuscript fingerprint</dt><dd><code>${esc(p.manuscript_hash)}</code></dd>
      ${p.memo ? `<dt>Organizer invoice memo (public on-chain)</dt><dd>${esc(p.memo)}</dd>` : ""}</dl>
    ${p.evidence.url ? `<a href="${esc(sourceUrl(p.evidence.url))}" target="_blank" rel="noopener">Organizer payment evidence</a>` : `<p class="origin">Human-specified demo invoice. This is not proof that the conference accepts Solana.</p>`}
    ${p.status === "awaiting_wallet" ? `<p class="origin">Expires ${esc(new Date(p.expires_at * 1000).toLocaleString())}. The app holds no private keys and cannot sign for you.${devnet ? " Switch Phantom to Solana devnet and fund it with test SOL before reviewing." : ""}</p>
      <button type="button" data-wallet-payment="${p.id}" ${busy ? "disabled" : ""}>${networkMismatch ? "Request belongs to a different network" : busy ? "Opening wallet…" : `Review ${esc(p.amount)} ${devnet ? "test SOL" : "SOL"} payment in Phantom`}</button>
      <button type="button" data-cancel-payment="${p.id}" ${busy ? "disabled" : ""}>Cancel unsigned request</button>` : ""}
    ${["wallet_requested", "submitted", "awaiting_wallet"].includes(p.status) ? `<form data-publication-form="verify-payment-${p.id}"><input type="hidden" name="payment_id" value="${p.id}">
      <label>Transaction signature<input type="text" name="signature" value="${esc(signature)}" required maxlength="88"></label><button type="submit">Verify finalized receipt</button></form>` : ""}
    ${p.verification_notice ? `<p class="origin">${esc(p.verification_notice)}</p>` : ""}
    ${p.status === "wallet_requested" ? `<p class="origin">Wallet review is in progress or was interrupted. Check your wallet history; do not send again. Paste a transaction signature above if one was submitted.</p>` : ""}
    ${p.receipt ? `<p>${devnet ? "Devnet transfer confirmed. Real conference registration has not been purchased." : "SOL transfer confirmed. The organizer must separately confirm your conference registration."}</p><a href="${esc(p.receipt.explorer)}" target="_blank" rel="noopener">View verified transaction</a>` : ""}
  </article>`;
}

async function publicationPost(path, body) {
  const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || "The request could not be completed.");
  return payload;
}

function publicationNotice(text) {
  state.trail.push({role: "agent", text});
  schedule();
}

async function payInWallet(paymentId) {
  if (pendingWallets.has(paymentId)) return;
  pendingWallets.add(paymentId);
  schedule();
  let claimed = false;
  let signature = null;
  const investigationId = state.id;
  const base = `/api/investigations/${investigationId}/payments/${paymentId}`;
  try {
    const provider = window.phantom?.solana;
    if (!provider?.isPhantom) throw new Error("Open this app in a browser with Phantom installed, or in Phantom’s browser. No payment was sent.");
    if (!window.solanaWeb3) throw new Error("The Solana SDK did not load. No payment was sent.");
    const connected = await provider.connect();
    const payer = connected.publicKey.toString();
    const request = await publicationPost(`${base}/wallet`, {payer});
    claimed = true;
    const intent = request.intent;
    const sdk = window.solanaWeb3;
    const instruction = sdk.SystemProgram.transfer({fromPubkey: new sdk.PublicKey(payer), toPubkey: new sdk.PublicKey(intent.recipient), lamports: intent.lamports});
    instruction.keys.push({pubkey: new sdk.PublicKey(intent.reference), isSigner: false, isWritable: false});
    if (intent.merchant_reference) instruction.keys.push({pubkey: new sdk.PublicKey(intent.merchant_reference), isSigner: false, isWritable: false});
    const transaction = new sdk.Transaction({feePayer: new sdk.PublicKey(payer), recentBlockhash: request.blockhash}).add(instruction);
    if (intent.memo) transaction.add(new sdk.TransactionInstruction({keys: [], programId: new sdk.PublicKey("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"), data: new TextEncoder().encode(intent.memo)}));
    const sent = await provider.signAndSendTransaction(transaction);
    signature = sent.signature;
    localStorage.setItem(`marginalia-payment-${paymentId}`, signature);
    await publicationPost(`${base}/submitted`, {signature});
    if (state.id === investigationId) {
      publicationNotice("The wallet submitted the transaction. Verify the finalized receipt below; do not pay again while it is pending.");
      await sendMargin({action: {type: "verify_payment", payment_id: paymentId, signature}});
    }
  } catch (error) {
    if (claimed && !signature && error.code === 4001) {
      await publicationPost(`${base}/rejected`, {code: 4001}).catch(() => {});
      publicationNotice("You rejected the wallet request. No transaction was submitted by this app.");
    } else {
      publicationNotice(signature ? `The wallet returned transaction ${signature}, but recording or verification failed. Use that signature to verify; do not pay again. ${error.message}`
        : `${error.message}${claimed ? " Check your wallet history before retrying; the request remains locked to prevent duplicate payments." : ""}`);
    }
  } finally {
    pendingWallets.delete(paymentId);
    schedule();
  }
}

desk.addEventListener("input", event => {
  const form = event.target.closest("form[data-publication-form]");
  if (form) form.setAttribute("data-dirty", "true");
});

desk.addEventListener("click", async event => {
  if (event.target.closest("[data-find-conferences]")) await sendMargin({action: {type: "find_conferences"}});
  const payment = event.target.closest("[data-wallet-payment]");
  if (payment) await payInWallet(payment.dataset.walletPayment);
  const cancel = event.target.closest("[data-cancel-payment]");
  if (cancel) await sendMargin({action: {type: "cancel_payment", payment_id: cancel.dataset.cancelPayment}});
});

desk.addEventListener("submit", async event => {
  const form = event.target.closest("form[data-publication-form]");
  if (!form) return;
  event.preventDefault();
  const id = form.getAttribute("data-publication-form");
  const data = Object.fromEntries(new FormData(form));
  for (const checkbox of form.querySelectorAll("input[type=checkbox]")) data[checkbox.name] = checkbox.checked;
  let action;
  if (id === "conference-source") action = {type: "find_conferences", source_url: data.source_url};
  else if (id.startsWith("select-conference-")) action = {type: "select_conference", ...data};
  else if (id === "create-manuscript") action = {type: "create_manuscript", ...data};
  else if (id === "save-manuscript") action = {type: "save_manuscript", ...data, version: Number(data.version)};
  else if (id === "review-manuscript") {
    if (desk.querySelector('form[data-publication-form="save-manuscript"][data-dirty="true"]')) { publicationNotice("Save manuscript edits before marking the current version reviewed."); return; }
    action = {type: "review_manuscript", ...data, version: Number(data.version)};
  } else if (id === "record-acceptance") action = {type: "record_acceptance", ...data};
  else if (id === "registration-invoice") action = {type: "prepare_registration", ...data, invoice_url: data.recipient ? `solana:${data.recipient.trim()}?amount=${encodeURIComponent(data.amount.trim())}` : data.invoice_url};
  else if (id.startsWith("verify-payment-")) action = {type: "verify_payment", ...data};
  if (!action) return;
  if (await sendMargin({action})) {
    const current = [...desk.querySelectorAll("form[data-publication-form]")].find(f => f.getAttribute("data-publication-form") === id);
    if (current) current.setAttribute("data-dirty", "false");
    form.setAttribute("data-dirty", "false");
    schedule();
  }
});
