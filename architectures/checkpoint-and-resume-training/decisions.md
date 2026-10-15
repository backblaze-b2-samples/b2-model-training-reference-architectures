# Decisions and tradeoffs

## ADR-1: Stage checkpoints locally, then upload
Status: accepted. Training continues as soon as the local write finishes, a network stall cannot corrupt the training process, and partial writes never reach B2. Tradeoff: needs local disk for at least one checkpoint per rank, and the newest checkpoint is not durable until its upload completes. Write directly with `FsspecWriter` when local disk is scarce and you accept slower checkpoints.

## ADR-2: Manifest written last is the commit marker
Status: accepted. B2 has no multi-object transactions. A checkpoint without `manifest.json` is incomplete by definition, and resume ignores it. Tradeoff: one extra object per checkpoint and a barrier before the coordinator writes it.

## ADR-3: Resume verifies checksums, not just presence
Status: accepted. Each object stores its SHA-256 in metadata and the manifest lists it. Resume refuses a checkpoint whose files do not match. Tradeoff: hashing on download costs CPU; it is small next to deserializing the checkpoint.

## ADR-4: Separate checkpoints and models buckets
Status: accepted. Checkpoints need short lifecycle rules and deletion. Models need Object Lock and replication. Those are bucket-level settings. Tradeoff: two buckets, and cross-bucket promotion needs a broader key or a download and upload. The sample uses one bucket with prefixes and per-object retention.

## ADR-5: Promote by copy, not by reference
Status: accepted. A model that only points at a checkpoint can be deleted by checkpoint pruning or lifecycle rules. A server-side copy needs no download and makes the model independent. Tradeoff: storage for the duplicate until the checkpoint is pruned.

## ADR-6: Governance-mode Object Lock for promoted models
Status: accepted. Protects against accidental deletes and leaked runtime keys, which lack `bypassGovernance`, while letting an administrator fix mistakes. Compliance mode cannot be shortened and is reserved for contractual or regulatory requirements. Enabling Object Lock on a bucket cannot be undone.

## ADR-7: Prune by deleting, purge by lifecycle
Status: accepted. A delete in a versioned bucket hides the file, and `daysFromHidingToDeleting` purges it later, which gives a short undo window. Avoid a blanket `daysFromUploadingToHiding` on active runs: it would hide the newest checkpoint of a long run.

## ADR-8: DCP as the default format
Status: accepted. Sharded saves scale with world size and can be loaded with a different world size. Tradeoff: more objects per checkpoint than a single `torch.save` file. Use safetensors for promoted inference artifacts when consumers need a framework-neutral format.
