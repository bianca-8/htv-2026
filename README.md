# Marginalia

A research desk shared by a person and an agent. The person owns the research question, selects and edits ideas, approves experiments, and interprets the results. The agent searches literature, proposes grounded hypotheses, runs approved computational work, and returns evidence for review.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). This is a local, single-user application; keep it bound to localhost.

The planner and experiment writer default to local Ollama (`llama3.1`). Start Ollama and pull that model, or set `OLLAMA_MODEL`. To use Gemini instead, set `EXPERIMENT_WRITER=gemini` and `GEMINI_API_KEY` (`GEMINI_MODEL` overrides `gemini-2.5-flash`). Remote model selection sends the approved source material and plans to that provider.

## Research workflow

1. **Literature search.** Enter a computational ML or computer-science question. Papers come from arXiv and, when configured, the Papiers library; OpenAlex is the public-index fallback. Titles, abstracts and source links come from those records. The query and search limitations remain visible. The application retrieves available arXiv HTML method/experiment excerpts for up to eight papers and retains complete abstracts in model context.
2. **Review papers.** Strike irrelevant papers, write guidance, and approve the kept sources. Nothing is executed during this step.
3. **Review ideas.** The planner synthesizes hypotheses, comparisons, ablations, reproductions or limitation tests. Each idea has a hypothesis, method, data, baseline, evaluation, assumptions, limitations, effort budget and source passages. Citations must match actual source text. A proposed hypothesis is explicitly distinct from a published finding. Empty searches are not proof of novelty. If model synthesis fails, editable evidence-based draft plans remain available; incomplete drafts cannot run.
4. **Choose and edit.** Select, defer or reject ideas independently, or add a human idea using kept paper IDs. Save edits before approval. Incomplete fallback suggestions require concrete inputs, methods and evaluation before they can run. “Draft a demonstration plan” prepares another editable plan and does not approve or execute it. Editing a fixed bench recipe converts it to a model-written experiment so the approved method is honored. Changes to papers, constraints, metrics, guidance or plans invalidate execution approval. An edit during model writing prevents the stale script from executing.
5. **Approve and execute.** Approval records immutable copies of the selected plan versions, seed, metric, precision floor, constraints and guidance. Each idea runs separately, with a baseline and its own artifacts. Execution never automatically follows a planner proposal. A failed idea does not fabricate a result or suppress the other approved ideas.
6. **Review results.** See measurements, baseline differences, data provenance, limitations, code, logs and the exact approved plan. Accept, request revision or reject the result, with a human interpretation. Prepare a rerun for fresh approval, or propose follow-up branches from a completed result. Prior results and review decisions remain saved.

General margin notes become explicit guidance for the next approval. Presets support no synthetic samples, no undersampling, no class weighting, no threshold tuning, metric selection and a validation precision floor. Existing run results retain the settings under which they were obtained; changing the objective does not silently alter them. Constraints can be removed using the buttons in the ideas panel.

**Pause** stops at workflow checkpoints. **Cancel active work** terminates an active experiment process and retains partial artifacts. A model request already in progress may finish in the background, but its stale output is not executed. Model planning/writing calls have a 90-second request timeout. **Redirect** starts another literature iteration, preserves the earlier branch/run history, and retains the standing question and constraints.

## Data, execution and interpretation

The reliable built-in executor covers tabular binary classification: logistic regression, random forests, gradient boosting, decision trees, SMOTE, random oversampling/undersampling, class weighting, threshold selection and EasyEnsemble, when supported by the kept papers. These are disclosed bench approximations, not exact paper reproductions. The baseline is an unmodified logistic regression on the same prepared input and seed-11 splits. Preprocessing is fit on training rows only; cutoffs use validation rows. Test bootstrap intervals do not refit models and do not capture training variability.

A trusted loader prepares a supported public dataset named in the kept literature, or a complete paper-stated normal synthesis. The prepared table is fingerprinted before approval and copied into each run. Numeric inputs are required; unsupported schemas and failed loads block execution without silent imputation. The smallest class is positive; multiclass tables are explicitly converted to one-versus-rest. A single table is currently shared by an investigation's ideas. To change the dataset, revise the source literature; arbitrary dataset upload and arbitrary repository reproduction are not supported.

Without prepared data, only a human-completed plan explicitly labeled **demonstration** can run using small inputs described in the plan. Demonstrations cannot establish benchmark performance. Incomplete methods or missing baselines must be supplied by the human; an agent may return insufficient information rather than invent them. General model-written experiments report numeric measurements and a baseline with matching measurement names, but schema validation does not establish scientific validity. Hypothesis support, experimental fidelity and reproduction success remain human review decisions.

Experiments use an OS sandbox: macOS `sandbox-exec`, or Linux `bubblewrap` (`bwrap`). Unsupported platforms fail closed. The sandbox denies networking and access to the user's documents and credentials; the child receives a minimal environment and can write only inside its run directory. Prepared inputs, scripts and approval manifests are protected from modification. Each process has a 180-second wall limit, 120-second CPU limit, file-size and artifact budgets; Linux also sets a 2 GB virtual memory limit. The macOS runner has no aggregate memory limit. Linux containment is implemented but the live containment suite has been verified on macOS only.

## Saved work and artifacts

By default, SQLite stores investigation snapshots and append-only event history in `.cache/research.sqlite3` (`MARGINALIA_DB` overrides it). Artifacts live in `experiments/<investigation-id>/<run-id>/`: approved plan, dependency versions, Python version, seed, script, logs, prepared input, result JSON and generated tables. The UI links to downloads and an investigation JSON export. Refreshing starts at “Open a saved investigation…”; choose an investigation from the selector to reopen it. Interrupted work restores without automatic execution and requires fresh approval. Redirected iterations remain in the export history.

Papiers library search is optional. Install and sign in to the `papiers` CLI (`PAPIERS_TOKEN`) to include your library. Library quotes must remain substrings of the saved text. Missing integration or an empty search does not cause invented papers.

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests -q
# Real OS containment checks, outside an already restrictive nested sandbox:
RUN_SANDBOX_TESTS=1 .venv/bin/python -m pytest tests/test_sandbox.py -q
```

The workflow tests cover selection-to-execution provenance, stale-plan cancellation, atomic edits, removed evidence, constraints, result validation, human review, rerun approval, persistent recovery and artifact path checks. The opt-in sandbox suite checks real NumPy/scikit-learn execution, private-file and network denial, cancellation, and a prepared-data experiment.

For a deterministic browser smoke check without calling a model provider, run `.venv/bin/python tests/ui_smoke_server.py` and open `http://127.0.0.1:8766`. This uses a separate temporary database and an explicitly labeled toy experiment.

## Conferences, manuscript drafts and Solana registration

After research ideas are proposed, a separate background search looks for conference calls. It uses Brave Search when `BRAVE_SEARCH_API_KEY` is configured, otherwise public DuckDuckGo/Bing search, with a sweep of official ML conference sites if public search is unavailable. You can also supply a particular official call-for-papers URL. Page text and model responses are treated as untrusted evidence: logistical details are displayed only as matching source passages. Explicit past deadline dates are flagged; ambiguous dates and timezone-specific cutoffs require review. Conference legitimacy, current edition, paper format and fit are human decisions, and a match does not predict acceptance. Public search sends the research question to the configured search provider.

The publication section creates an editable Markdown draft from completed experiments you marked **accepted**. It includes the actual measurements, baseline, data description and reproducibility references. The introduction and discussion need author writing; this is a research draft, not a submission-ready publication or a claim of new findings. Save and review the current version, select a conference, then record its acceptance/submission reference. The app does not submit manuscripts, accept legal terms, fabricate acceptance, or contact organizers.

Payments are **registration fees after acceptance**, using **Solana mainnet and real SOL by default** (`SOLANA_NETWORK=mainnet-beta`). No private key or seed phrase is entered into this application. Developers can explicitly select `SOLANA_NETWORK=devnet` for tests; test transfers do not purchase registration.

The payment button connects the injected **Phantom** Solana provider, constructs a native SOL transfer using the locally bundled `@solana/web3.js` 1.98.4, and asks the wallet to sign/send it. It shows the recipient, exact amount, network and manuscript fingerprint before wallet approval. The app never signs or holds keys. Wallet rejection is handled, submitted signatures are retained for recovery, and an already claimed/submitted request cannot be automatically paid again. Ambiguous wallet errors leave the request locked: check wallet history and verify its signature rather than resending.

Server-side receipt checks require a finalized successful transaction with the correct recipient, exact lamports, signer and unique invoice reference (and merchant memo/reference when supplied). An RPC genesis check prevents a devnet label from being paired with a mainnet endpoint. `SOLANA_RPC_URL` may configure a dedicated operator-controlled endpoint; its network must match `SOLANA_NETWORK`. A submitted signature alone is never marked paid. Manuscript edits or withdrawing acceptance of a supporting result block stale payment requests. Payment requests, manuscript versions, conference sources and receipt history persist with the investigation.

Mainnet is selected automatically unless the operator overrides `SOLANA_NETWORK`. The organizer must issue a fixed **native SOL Solana Pay transfer URL** on a public payment page, and you must confirm it is the official organizer. The server checks that page advertises the exact invoice. Card-only conferences, arbitrary currency conversion, custodial spending, SPL/USDC payments and interactive Solana Pay transaction-request URLs are not supported. An on-chain transfer is distinct from the organizer confirming registration; retain the organizer's receipt too.

Publication/payment tests use mocked RPC responses and no real funds. Run `.venv/bin/python -m pytest tests/test_publication.py -q`. With Node installed, `node tests/test_wallet.mjs` checks the unsigned transfer, memo, reference and signer without connecting a wallet or broadcasting anything. The deterministic UI fixture includes clearly labeled synthetic conference data for interface testing only.

During literature search and paper review, use **Add a paper by arXiv link** to import an abstract, PDF, or HTML URL. The server retrieves the real arXiv metadata, deduplicates paper versions, and includes the paper in the human-approved literature set. Other link hosts are currently unsupported.

## Vultr Managed PostgreSQL

The app uses PostgreSQL when `DATABASE_URL` or `TIGER_DATABASE_URL` is set, and local SQLite otherwise. PostgreSQL stores investigation snapshots and event payloads as JSONB; saves update the snapshot and append its event in one transaction. A configured PostgreSQL failure stops the operation rather than writing to a separate local database.

1. Create a **Managed PostgreSQL** database in Vultr, preferably in the backend's region. Copy the host, port, database name, and credentials from its connection panel. Download its signed CA certificate. See [Vultr's connection instructions](https://docs.vultr.com/products/storage/databases/postgresql/management/connection/connection-details).
2. Permit your backend's outbound IP in the database's trusted sources. Keep the database credentials on the server.
3. Install `requirements.txt`. Copy `.env.example` to `.env`, uncomment and fill in `DATABASE_URL` and `PGSSLROOTCERT`. URL-encode special characters in the connection URL's username/password. Use an absolute CA file path. The app enforces TLS with certificate and hostname verification (`verify-full`), including when a URL requests a weaker SSL mode.
4. Start with the environment file:

   ```bash
   .venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000 --env-file .env
   ```

   For this command install the included `python-dotenv` dependency. Other launchers can inject the two environment variables directly. Python itself does not automatically load `.env`.

To transfer existing SQLite investigations, stop the backend first, retain a backup, and run against an **empty** PostgreSQL target. With your configured `.env`:

```bash
.venv/bin/python -m dotenv -f .env run -- .venv/bin/python -m backend.migrate_postgres --source .cache/research.sqlite3
```

The migration is atomic, preserves IDs and event ordering, refuses a nonempty target, and leaves SQLite untouched. Restart with the PostgreSQL configuration and open a saved investigation to verify it. To test a disposable PostgreSQL database, set `TEST_POSTGRES_URL` and `PGSSLROOTCERT`, then run `.venv/bin/python -m pytest tests/test_storage_postgres.py -q`; the integration test creates and drops its own isolated schema.

This adds database connectivity, not Vultr account provisioning. Experiment files still live in `experiments/` and need persistent disk storage. The backend remains a single-user, single-process app: run one worker. PostgreSQL alone does not add authentication, shared job coordination, or safe public multi-user hosting.

## Tiger Data / Tiger Cloud

Tiger Cloud works with the same PostgreSQL storage adapter. Use it as an alternative to Vultr: each app instance writes to one database, not both. Your app can run on Vultr while the database runs on Tiger Cloud. No TimescaleDB or vector extension is required for saved investigations and event history.

1. Create or choose a Tiger Cloud service and wait until it is **Ready**. Copy the PostgreSQL URL from the service's connection panel and include the database password. See [Tiger Data connection details](https://www.tigerdata.com/docs/integrate/find-connection-details).
2. In your ignored `.env`, set `TIGER_DATABASE_URL` using the example in `.env.example`, with your exact host, port and database name. Alternatively use `DATABASE_URL`. If both are set to different URLs, startup fails rather than choosing the wrong database.
3. TLS certificate and hostname verification is enforced even if the supplied URL says `sslmode=require`. Public CA certificates use the bundled trust store. Remove an old Vultr `PGSSLROOTCERT` when switching providers, or replace it with the CA certificate your Tiger service requires. An explicit URL `sslrootcert` is also supported; `PGSSLROOTCERT` takes precedence.
4. Start the app using the same `uvicorn --env-file .env` command above. To import local investigations into an empty Tiger database, use the same SQLite migration command above. Migration leaves the source intact; it does not transfer an existing Vultr PostgreSQL database.

If you prefer the Tiger CLI from the supplied guide, install and authenticate it in your own interactive terminal:

```bash
brew install --cask timescale/tap/tiger-cli
tiger auth login
tiger service list
```

Choose an existing service in Tiger Console, or create one there and copy its connection details. Tiger CLI authentication manages cloud resources; the running application still needs the PostgreSQL URL and database password. The app does not invoke Tiger CLI, provision paid services, or expose database management to the research agent.
