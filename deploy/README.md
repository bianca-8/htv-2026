# Deploy a protected, shared demo on Vultr (Ubuntu 24.04)

Use one server and one app worker. This is a shared research desk behind a password, not a multi-user account system. Use Gemini for model generation; this guide does not install Ollama. Tiger Data is optional; local SQLite is sufficient to start.

## 1. DNS and connect

Keep Porkbun nameservers. Add an A record with blank Host and your Vultr public IPv4 as Answer. Add a CNAME with Host `www` and your bare domain as Answer. Replace conflicting parking records for these hosts; keep email records. Do not add an AAAA record unless IPv6 is configured too.

On your Mac, open Terminal. Replace `YOUR_SERVER_IP`:

```bash
ssh root@YOUR_SERVER_IP
```

Use the server's initial password from the Vultr instance panel if you did not select an SSH key. If you chose a limited-login image, use its assigned username and prefix administrative commands below with `sudo`. Confirm a first-connect host key only after comparing its fingerprint through the Vultr console. Keep this SSH session open while configuring firewall access.

## 2. Install and copy the code

Run on the Ubuntu server, as root:

```bash
apt update
apt install -y git python3-venv python3-pip bubblewrap ufw curl gnupg debian-keyring debian-archive-keyring apt-transport-https
adduser --system --group --home /var/lib/marginalia marginalia
git clone https://github.com/bianca-8/htv-2026.git /opt/marginalia
chown -R marginalia:marginalia /opt/marginalia
sudo -u marginalia /usr/bin/python3 -m venv /opt/marginalia/.venv
sudo -u marginalia /opt/marginalia/.venv/bin/pip install -r /opt/marginalia/requirements.txt
install -d -o marginalia -g marginalia -m 700 /opt/marginalia/.cache /opt/marginalia/experiments
```

The repository must contain your latest app and `deploy/` files first. For a private repository, authenticate using a read-only deploy key or GitHub authentication; do not put an access token into the clone URL. If `/opt/marginalia` already contains an installation, do not clone over it—update that checkout instead.

## 3. Configure secrets and verify execution

```bash
install -m 600 /opt/marginalia/deploy/server.env.example /etc/marginalia.env
nano /etc/marginalia.env
```

Replace the Gemini key placeholder. Optionally uncomment and fill `TIGER_DATABASE_URL`. URL-encode special characters in database credentials. The service reads this file; secrets stay outside the repository. For Vultr's custom CA, put the certificate in `/opt/marginalia/certs/`, make it readable by `marginalia`, and set `PGSSLROOTCERT`.

```bash
sudo -u marginalia /opt/marginalia/.venv/bin/python /opt/marginalia/deploy/check_sandbox.py
```

Require a passing check before running generated code. Ubuntu policies may restrict unprivileged user namespaces; if the check fails, inspect the error and resolve the bubblewrap/AppArmor setup. Do not globally disable security restrictions or bypass the sandbox.

## 4. Start the app automatically

```bash
install -m 644 /opt/marginalia/deploy/marginalia.service /etc/systemd/system/marginalia.service
systemctl daemon-reload
systemctl enable --now marginalia
systemctl status marginalia --no-pager
curl -I http://127.0.0.1:8000/
```

A running service and HTTP 200 confirm startup. If the database cannot connect, check its credentials, certificate, and allowed source IPs:

```bash
journalctl -u marginalia -n 60 --no-pager
```

## 5. HTTPS and password protection

Install current Caddy from its official repository, following https://caddyserver.com/docs/install#debian-ubuntu-raspbian:

```bash
curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt -o /etc/apt/sources.list.d/caddy-stable.list
apt update
apt install -y caddy
caddy hash-password
```

Enter a new demo login password at the prompt; retain the resulting hash. Then:

```bash
install -m 644 /opt/marginalia/deploy/Caddyfile.example /etc/caddy/Caddyfile
nano /etc/caddy/Caddyfile
```

Replace `example.com` and `www.example.com` with your domains. Replace the password hash placeholder with the full hash, including its `$` characters. The login username is `demo`. The password and hash are different; use the original password to log in. Save with Ctrl+O, Enter, then Ctrl+X.

```bash
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
```

## 6. Firewall, then enable the public website

In this SSH session, find the IP used for your connection:

```bash
printenv SSH_CONNECTION
```

The first value is your client IP. Replace `YOUR_CLIENT_IP` below with that value. If it is IPv6, it must be the full IPv6 address. Do not use the server IP for this rule.

```bash
ufw allow from YOUR_CLIENT_IP to any port 22 proto tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
ufw status
```

Keep the existing session open and test a second SSH connection from your Mac before closing it. If a Vultr Firewall Group is attached, permit the same traffic there too. Port 8000 stays bound to localhost and should not be opened publicly.

Once Porkbun DNS resolves to this server:

```bash
systemctl reload caddy
```

Open `https://YOUR_DOMAIN`. You should receive a login prompt; log in as `demo` using the original password. Caddy obtains and renews HTTPS certificates automatically. Verify the browser connection is HTTPS, paper import works, and only the branch you approve runs. Solana still requires wallet approval and an authentic organizer invoice; do not fabricate conference acceptance for the demo.

## Updates and persistence

GitHub stores the source. Cloning does not automatically deploy future commits. To update the existing checkout after pushing changes:

```bash
sudo -u marginalia git -C /opt/marginalia pull --ff-only
sudo -u marginalia /opt/marginalia/.venv/bin/pip install -r /opt/marginalia/requirements.txt
systemctl restart marginalia
```

Stop active experiments before updates. Local SQLite is in `.cache/`; run artifacts are in `experiments/`. Retain backups of both even when using a cloud database. To migrate existing local SQLite history, use the main README's migration procedure before starting against the new empty PostgreSQL database.
