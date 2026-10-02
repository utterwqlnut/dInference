import * as anchor from "@coral-xyz/anchor";
import { PublicKey, Keypair, LAMPORTS_PER_SOL, SystemProgram } from "@solana/web3.js";
import { createHash } from "crypto";
import BN from "bn.js";

export const SEEDS = {
  config:    Buffer.from("config"),
  vault:     Buffer.from("vault"),
  rotating:  Buffer.from("seed"),
  model:     Buffer.from("model"),
  queue:     Buffer.from("queue"),
  provider:  Buffer.from("provider"),
  verifier:  Buffer.from("verifier"),
  request:   Buffer.from("req"),
  response:  Buffer.from("resp"),
  challenge: Buffer.from("chal"),
  avail:     Buffer.from("avail"),
};

export function sha256(input: Buffer | string): Buffer {
  return createHash("sha256").update(input).digest();
}

export function pda(programId: PublicKey, seeds: Buffer[]): PublicKey {
  return PublicKey.findProgramAddressSync(seeds, programId)[0];
}

export function configPda(programId: PublicKey) {
  return pda(programId, [SEEDS.config]);
}
export function vaultPda(programId: PublicKey) {
  return pda(programId, [SEEDS.vault]);
}
export function rotatingSeedPda(programId: PublicKey) {
  return pda(programId, [SEEDS.rotating]);
}
export function modelPda(programId: PublicKey, modelId: string) {
  return pda(programId, [SEEDS.model, sha256(modelId)]);
}
export function queuePda(programId: PublicKey, modelId: string) {
  return pda(programId, [SEEDS.queue, sha256(modelId)]);
}
export function poolPda(programId: PublicKey, modelId: string) {
  return pda(programId, [SEEDS.avail, sha256(modelId)]);
}
export function providerPda(programId: PublicKey, operator: PublicKey) {
  return pda(programId, [SEEDS.provider, operator.toBuffer()]);
}
export function verifierPda(programId: PublicKey, operator: PublicKey) {
  return pda(programId, [SEEDS.verifier, operator.toBuffer()]);
}
export function requestPda(programId: PublicKey, user: PublicKey, nonce: BN) {
  return pda(programId, [SEEDS.request, user.toBuffer(), nonce.toArrayLike(Buffer, "le", 8)]);
}
export function responsePda(programId: PublicKey, request: PublicKey) {
  return pda(programId, [SEEDS.response, request.toBuffer()]);
}
export function challengePda(programId: PublicKey, request: PublicKey) {
  return pda(programId, [SEEDS.challenge, request.toBuffer()]);
}

export async function airdrop(
  connection: anchor.web3.Connection,
  pubkey: PublicKey,
  sol = 10,
) {
  const sig = await connection.requestAirdrop(pubkey, sol * LAMPORTS_PER_SOL);
  await connection.confirmTransaction(sig, "confirmed");
}

export async function balance(
  connection: anchor.web3.Connection,
  pubkey: PublicKey,
): Promise<number> {
  return await connection.getBalance(pubkey);
}

export function defaultConfigArgs(initialSeed: Buffer): any {
  return {
    minProviderStake: new BN(1_000_000),
    minVerifierStake: new BN(500_000),
    minUserBond:      new BN(10_000),
    challengeWindowSlots: new BN(100),
    voteWindowSlots:      new BN(20),
    unstakeCooldownSlots: new BN(150),
    seedRotationSlots:    new BN(50),
    responseTimeoutSlots: new BN(30),
    cosThreshold: 9900,
    verifierCommitteeSize: 3,
    quorumFraction: 6700,
    tokSMinThreshold: 5,
    maxNewTokens: 128,
    initialSeed: Array.from(initialSeed),
  };
}

export function makeFingerprint(dim = 64, fill = 0.5): number[] {
  return new Array(dim).fill(fill);
}

export function randomTxid(): Buffer {
  return Buffer.from(Array.from({ length: 32 }, () => Math.floor(Math.random() * 256)));
}
