# Bare-metal conversion runbook: pve13 & proxmox-mini → k3s servers

Converts the two weakest Proxmox hosts into bare-metal k3s control-plane nodes,
reclaiming the virtualization overhead and giving etcd the full box. The Trigkey
(`pve`) stays on Proxmox (it hosts HAOS, kube-leader-2, homebridge, and the
secondary pihole LXC).

**End state:** `pve` is a *standalone* Proxmox node (corosync cluster dissolved);
pve13 and the mini run Debian + k3s servers directly.

| Host (now) | Mgmt IP | Guest being replaced | Bare-metal role | Node IP (VLAN 8) |
|---|---|---|---|---|
| pve13 | 192.168.1.13 | VM 106 `kube-leader` (cp/etcd) | k3s server | 192.168.8.23 |
| proxmox-mini | 192.168.1.10 | VM 127 `kube-worker-1` (cp/etcd) | k3s server | 192.168.8.21 |

Do these **one host at a time**, fully completing and verifying one before starting
the next. They are 2 of the 3 etcd members — never have two of them out at once.

---

## Prerequisites (all DONE as of 2026-07-20 — verify before starting)

- [x] All agents' `K3S_URL` point at the kube-vip VIP `https://192.168.8.10:6443`, not any single node's IP. (verify: `grep K3S_URL /etc/systemd/system/k3s-agent.service.env` on any agent)
- [x] kube-vip DaemonSet auto-detects its interface (no hardcoded `ens18`) — so a bare-metal node can host the VIP. **Requirement:** the new host's default route must be on its VLAN-8 interface (see networking below).
- [ ] **Pin the install to whatever the cluster is running RIGHT NOW — this drifts.**
      Check first: `kubectl get nodes -o wide` and use that exact version.
      (Was v1.34.6+k3s1 when written; **v1.36.2+k3s1** as of 2026-07-21.)
      Joining at the wrong version is the easiest way to break this.
- [x] etcd snapshot copied off-cluster (Beelink `/mnt/usb/cluster-backups/`), plus vzdump of the VM on nfs-backup.
- [x] Synology RAID healthy (nfs-backup target has redundancy again).
- [ ] Take a fresh etcd snapshot the day of: `k3s etcd-snapshot save --name pre-convert-<host>` on kube-leader-2, copy it off-box.
- [ ] Confirm cluster is fully healthy first: all nodes Ready, etcd 3/3, no failing ArgoCD apps.

---

## Networking for the bare-metal host (important)

The VMs being replaced live on **VLAN 8 only** (single NIC, `tag=8`, default route via
`192.168.8.1`). Match that so kube-vip's auto-detect binds the VIP to the right interface.

Keep the switch port as-is (it already trunks VLAN 8: untagged = 192.168.1.x for the
old Proxmox mgmt, tagged = VLAN 8). The Debian installer will DHCP on the **untagged**
side and come up as 192.168.1.x — that's fine for installing; you add the VLAN-8
interface afterwards and make it primary.

**Debian uses `ifupdown`, not netplan** (netplan is Ubuntu). Find the real NIC name
with `ip -br link`, then `/etc/network/interfaces`:

```
auto lo
iface lo inet loopback

# physical NIC: no IP of its own
auto enp2s0
iface enp2s0 inet manual

# VLAN 8 = the cluster network, and the default route lives here
auto enp2s0.8
iface enp2s0.8 inet static
    address 192.168.8.21/24        # .23 for pve13
    gateway 192.168.8.1
    dns-nameservers 192.168.8.204 192.168.8.254
    vlan-raw-device enp2s0
```

Needs the `vlan` package: `sudo apt-get install -y vlan` (loads 8021q).
Apply with `sudo systemctl restart networking` — or reboot, which is safer since
you're changing the interface you're connected over.

Manage/SSH the host via its VLAN-8 IP (192.168.8.23 / .21) — reachable from the Mac.
Verify before installing k3s: `ip route | grep default` shows the route on `enp2s0.8`,
and `ping 192.168.8.1` works.

---

## Per-host conversion procedure

### 1. Remove the node from Kubernetes (do NOT skip — clean etcd removal)
```bash
# from a machine with cluster admin (or ssh to a server + sudo k3s kubectl)
kubectl drain <kube-leader|kube-worker-1> --ignore-daemonsets --delete-emptydir-data --timeout=300s
kubectl delete node <kube-leader|kube-worker-1>   # k3s managed-etcd controller removes the etcd member
# confirm etcd is now 2/2 and healthy before continuing:
kubectl get nodes            # the node is gone
# on a surviving server: k3s etcd-snapshot ls, and check `etcdctl member list` shows 2 members
```
You are now on **2/3 etcd — quorate but zero fault tolerance.** Do not reboot the mini
or Trigkey until this host is fully back.

### 2. Shut down (don't destroy yet) the guest — it's the rollback
```bash
# on the Proxmox host
qm shutdown 106     # pve13:  kube-leader
qm shutdown 127     # mini:   kube-worker-1
```
Leave the VM defined until the bare-metal node is proven healthy.

### 3. Remove the host from the Proxmox cluster
Run from a **surviving, quorate** Proxmox node (not the one being wiped):
```bash
# e.g. from pve
pvecm delnode pve13        # (or proxmox-mini for the second pass)
pvecm status               # confirm the node count dropped and it's still Quorate
```
- After removing **pve13**: cluster is 2 nodes (mini + pve). Quorum now needs *both* up.
- Before removing the **mini** (second pass): you'd be about to drop to 1 node, which is
  non-quorate by default. **This is where the cluster gets dissolved — see step 6.**
- Optional cleanup on survivors: `rm -rf /etc/pve/nodes/<host>` if a stale dir lingers.

### 4. Wipe & install Debian 13, then bootstrap
Install Debian 13, apply the networking above, then the standard node bootstrap
(see CLAUDE.md "Node Bootstrap Requirements"):
Use **Debian 13 (trixie)** — matches the Pis and the existing kube-worker-1, and it's
the same base Proxmox already ran on this hardware (proven boot/EFI/NIC support).
```bash
sudo apt-get update
sudo apt-get install -y containernetworking-plugins nfs-common cifs-utils open-iscsi
sudo cp /usr/lib/cni/* /opt/cni/bin/          # flannel runs as a DaemonSet; base CNI plugins must exist
```
Conntrack check (x86 trixie also runs the 6.12 kernel that needs the RPi fix on some
NICs — verify, and apply if you see DNS/UDP flakiness after join):
```bash
sudo sysctl net.netfilter.nf_conntrack_checksum          # want 0
# if 1: echo 'net.netfilter.nf_conntrack_checksum=0' | sudo tee /etc/sysctl.d/99-conntrack.conf && sudo sysctl -w net.netfilter.nf_conntrack_checksum=0
```

### 5. Join as a k3s **server**, matching the existing server config
The existing servers run with a specific config (flannel-backend=none, disables, etc.).
**Copy the authoritative flags from a live server rather than guessing:**
```bash
# on kube-leader-2 (or any current server):
sudo cat /etc/rancher/k3s/config.yaml         # capture this
sudo cat /var/lib/rancher/k3s/server/token    # capture the server token
```
On the new host, place that same `/etc/rancher/k3s/config.yaml`, then:
```bash
curl -sfL https://get.k3s.io | \
  INSTALL_K3S_VERSION=<CURRENT CLUSTER VERSION - verify with kubectl get nodes> \
  K3S_TOKEN='<server-token>' \
  sh -s - server \
    --server https://192.168.8.10:6443 \
    --node-ip 192.168.8.23              # .21 for the mini
# (flannel-backend=none, disables, etc. come from config.yaml)
```
Node name defaults to the hostname — set the hostname to `kube-leader` / `kube-worker-1`
before install if you want to keep the existing names (labels/refs stay valid).

### 6. Dissolve the Proxmox cluster (during the MINI pass only)
After the mini leaves, `pve` is the only Proxmox host. Don't limp along on
`pvecm expected 1` — make it a clean standalone node:
```bash
# on pve, after `pvecm delnode proxmox-mini`:
systemctl stop pve-cluster corosync
pmxcfs -l                               # start pmxcfs in local mode
rm /etc/pve/corosync.conf
rm -rf /etc/corosync/*
killall pmxcfs
systemctl start pve-cluster             # now standalone, no corosync
# verify: `pvecm status` reports no cluster; VMs start normally; /etc/pve is writable
```
(Reference: Proxmox "Separate a node without reinstalling" procedure.)

---

## Verification (after each host)
- `kubectl get nodes` → new node Ready, roles `control-plane,etcd`, version v1.34.6+k3s1.
- etcd back to 3 members, all healthy: on a server, `k3s etcdctl member list` / endpoint health.
- kube-vip: VIP still answers (`curl -sk https://192.168.8.10:6443/ping` → pong); the new
  node's kube-vip pod is Running.
- Workloads schedulable on it (or, if you taint control-plane nodes later, that they drain cleanly).
- `pvecm status` on remaining Proxmox node(s) is Quorate (or standalone after the mini pass).

## Rollback
Nothing is destroyed until you're confident:
- The VM is only shut down, not deleted — power it back on to restore the old member
  (its etcd state is stale, so you'd delete the half-built bare-metal node first, then
  boot the VM and let it re-sync as the member it was).
- 2/3 etcd keeps the cluster fully functional throughout, so a failed install is not an
  outage — just retry. The etcd snapshot is the disaster-only net.

## Post-conversion cleanup
- Delete the old VMs (`qm destroy 106` / `127`) and the mini's stopped `mariadb` LXC (100) once stable.
- Remove the retired host IPs from the `pve` scrape job in `core/prometheus.yaml` if the
  Proxmox exporter targets change (192.168.1.10 / .13 stop being Proxmox).
- Update `infra/terraform` — these hosts are no longer Terraform-managed VMs.
- Keep the pre-conversion vzdumps ~1 week, then let retention prune them.
