/**
 * End-to-end integration tests for the dInference Anchor program.
 *
 * Prerequisites:
 *   - solana-test-validator running on localhost
 *   - `anchor build && anchor deploy` has placed the program on the validator
 *   - IDL is written to target/idl/dinference.json
 *
 * Run: `anchor test` (which spins up a validator automatically) or:
 *      `npm test` against a pre-running validator.
 */

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { Keypair, PublicKey, LAMPORTS_PER_SOL } from "@solana/web3.js";
import { expect } from "chai";
import BN from "bn.js";

import {
  airdrop, balance,
  configPda, vaultPda, rotatingSeedPda,
  modelPda, queuePda, poolPda,
  providerPda, verifierPda,
  requestPda, responsePda, challengePda,
  defaultConfigArgs, makeFingerprint, randomTxid, sha256,
} from "./helpers";

// Autoloaded by `anchor test` or manually generated.
// For now we type as `any` since the IDL is only present post-build.
let program: Program<any>;
let provider: anchor.AnchorProvider;
let programId: PublicKey;

const MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct";
const HIDDEN_DIM = 1536;

describe("dInference", () => {
  // Actors (keypairs created once per suite).
  const admin           = Keypair.generate();
  const providerOp      = Keypair.generate();
  const verifier1       = Keypair.generate();
  const verifier2       = Keypair.generate();
  const verifier3       = Keypair.generate();
  const user            = Keypair.generate();
  const otherUser       = Keypair.generate();

  before(async () => {
    provider = anchor.AnchorProvider.env();
    anchor.setProvider(provider);
    program = anchor.workspace.Dinference as Program<any>;
    programId = program.programId;

    // Fund all actors.
    for (const k of [admin, providerOp, verifier1, verifier2, verifier3, user, otherUser]) {
      await airdrop(provider.connection, k.publicKey, 100);
    }
  });

  // -----------------------------------------------------------------------
  //  Initialization
  // -----------------------------------------------------------------------

  describe("init", () => {
    it("init_config sets protocol parameters and creates vault", async () => {
      const args = defaultConfigArgs(sha256("initial-seed"));

      await program.methods
        .initConfig(args)
        .accounts({
          admin: admin.publicKey,
          config: configPda(programId),
          rotatingSeed: rotatingSeedPda(programId),
          stakeVault: vaultPda(programId),
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .signers([admin])
        .rpc();

      const cfg = await program.account.globalConfig.fetch(configPda(programId));
      expect(cfg.admin.toBase58()).to.equal(admin.publicKey.toBase58());
      expect(cfg.cosThreshold).to.equal(9900);
      expect(cfg.verifierCommitteeSize).to.equal(3);
      expect(cfg.maxNewTokens).to.equal(128);
    });

    it("init_model registers model + queue + pool PDAs", async () => {
      await program.methods
        .initModel(MODEL_ID, HIDDEN_DIM)
        .accounts({
          admin: admin.publicKey,
          config: configPda(programId),
          model: modelPda(programId, MODEL_ID),
          queue: queuePda(programId, MODEL_ID),
          pool: poolPda(programId, MODEL_ID),
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .signers([admin])
        .rpc();

      const model = await program.account.modelInfo.fetch(modelPda(programId, MODEL_ID));
      expect(model.modelId).to.equal(MODEL_ID);
      expect(model.allowed).to.be.true;
      expect(model.activeProviders).to.equal(0);

      const queue = await program.account.requestQueue.fetch(queuePda(programId, MODEL_ID));
      expect(queue.pending.length).to.equal(0);
    });

    it("non-admin cannot init_model", async () => {
      const randomUser = Keypair.generate();
      await airdrop(provider.connection, randomUser.publicKey, 10);
      try {
        await program.methods
          .initModel("some-other-model", 512)
          .accounts({
            admin: randomUser.publicKey,
            config: configPda(programId),
            model: modelPda(programId, "some-other-model"),
            queue: queuePda(programId, "some-other-model"),
            systemProgram: anchor.web3.SystemProgram.programId,
          })
          .signers([randomUser])
          .rpc();
        expect.fail("should have reverted");
      } catch (e: any) {
        expect(e.toString()).to.match(/AdminOnly|constraint/i);
      }
    });
  });

  // -----------------------------------------------------------------------
  //  Registration & staking
  // -----------------------------------------------------------------------

  describe("registration", () => {
    it("register_provider transfers SOL to vault and creates provider account", async () => {
      const stake = new BN(2_000_000);
      const vaultBefore = await balance(provider.connection, vaultPda(programId));

      await program.methods
        .registerProvider(MODEL_ID, stake)
        .accounts({
          operator: providerOp.publicKey,
          config: configPda(programId),
          model: modelPda(programId, MODEL_ID),
          provider: providerPda(programId, providerOp.publicKey),
          stakeVault: vaultPda(programId),
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .signers([providerOp])
        .rpc();

      const p = await program.account.providerAccount.fetch(providerPda(programId, providerOp.publicKey));
      expect(p.operator.toBase58()).to.equal(providerOp.publicKey.toBase58());
      expect(p.stakeLamports.toNumber()).to.equal(2_000_000);
      expect(p.isActive).to.be.true;
      expect(p.runningTokS).to.equal(0);

      const vaultAfter = await balance(provider.connection, vaultPda(programId));
      expect(vaultAfter - vaultBefore).to.equal(2_000_000);

      const model = await program.account.modelInfo.fetch(modelPda(programId, MODEL_ID));
      expect(model.activeProviders).to.equal(1);
    });

    it("register_provider rejects stake below minimum", async () => {
      const sketchy = Keypair.generate();
      await airdrop(provider.connection, sketchy.publicKey, 10);
      try {
        await program.methods
          .registerProvider(MODEL_ID, new BN(100)) // way below min
          .accounts({
            operator: sketchy.publicKey,
            config: configPda(programId),
            model: modelPda(programId, MODEL_ID),
            provider: providerPda(programId, sketchy.publicKey),
            stakeVault: vaultPda(programId),
            systemProgram: anchor.web3.SystemProgram.programId,
          })
          .signers([sketchy])
          .rpc();
        expect.fail("should have reverted");
      } catch (e: any) {
        expect(e.toString()).to.match(/StakeTooLow/);
      }
    });

    it("register_verifier works for multiple verifiers", async () => {
      for (const v of [verifier1, verifier2, verifier3]) {
        await program.methods
          .registerVerifier(MODEL_ID, new BN(600_000))
          .accounts({
            operator: v.publicKey,
            config: configPda(programId),
            model: modelPda(programId, MODEL_ID),
            verifier: verifierPda(programId, v.publicKey),
            stakeVault: vaultPda(programId),
            systemProgram: anchor.web3.SystemProgram.programId,
          })
          .signers([v])
          .rpc();
      }

      for (const v of [verifier1, verifier2, verifier3]) {
        const acct = await program.account.verifierAccount.fetch(verifierPda(programId, v.publicKey));
        expect(acct.isActive).to.be.true;
        expect(acct.stakeLamports.toNumber()).to.equal(600_000);
      }
    });
  });

  // -----------------------------------------------------------------------
  //  Request lifecycle
  // -----------------------------------------------------------------------

  describe("request lifecycle", () => {
    const nonce = new BN(1);
    const promptTxid = randomTxid();
    const responseTxid = randomTxid();

    it("submit_request escrows payment and queues the request", async () => {
      const vaultBefore = await balance(provider.connection, vaultPda(programId));
      const payment = new BN(5_000_000);
      const priorityFee = new BN(200_000);

      await program.methods
        .submitRequest(nonce, Array.from(promptTxid), payment, priorityFee)
        .accounts({
          user: user.publicKey,
          config: configPda(programId),
          model: modelPda(programId, MODEL_ID),
          queue: queuePda(programId, MODEL_ID),
          rotatingSeed: rotatingSeedPda(programId),
          request: requestPda(programId, user.publicKey, nonce),
          stakeVault: vaultPda(programId),
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .signers([user])
        .rpc();

      const req = await program.account.inferenceRequest.fetch(
        requestPda(programId, user.publicKey, nonce)
      );
      expect(req.user.toBase58()).to.equal(user.publicKey.toBase58());
      expect(req.paymentLamports.toNumber()).to.equal(5_000_000);
      expect(req.priorityFee.toNumber()).to.equal(200_000);
      expect(Buffer.from(req.promptTxid)).to.deep.equal(promptTxid);
      expect(req.status).to.deep.equal({ pending: {} });

      const vaultAfter = await balance(provider.connection, vaultPda(programId));
      expect(vaultAfter - vaultBefore).to.equal(5_200_000);

      const queue = await program.account.requestQueue.fetch(queuePda(programId, MODEL_ID));
      expect(queue.pending.length).to.equal(1);
    });

    it("assign_next pops the FIFO head and sets the provider", async () => {
      await program.methods
        .assignNext()
        .accounts({
          caller: providerOp.publicKey,
          queue: queuePda(programId, MODEL_ID),
          request: requestPda(programId, user.publicKey, nonce),
          provider: providerPda(programId, providerOp.publicKey),
        })
        .signers([providerOp])
        .rpc();

      const req = await program.account.inferenceRequest.fetch(
        requestPda(programId, user.publicKey, nonce)
      );
      expect(req.status).to.deep.equal({ assigned: {} });
      expect(req.assignedProvider.toBase58()).to.equal(providerOp.publicKey.toBase58());

      const queue = await program.account.requestQueue.fetch(queuePda(programId, MODEL_ID));
      expect(queue.pending.length).to.equal(0);
    });

    it("submit_response commits fingerprint and updates provider EMA", async () => {
      const fingerprint = makeFingerprint(64, 0.42);
      const nTokens = 100;

      await program.methods
        .submitResponse(Array.from(responseTxid), fingerprint, nTokens)
        .accounts({
          providerOperator: providerOp.publicKey,
          config: configPda(programId),
          provider: providerPda(programId, providerOp.publicKey),
          request: requestPda(programId, user.publicKey, nonce),
          commitment: responsePda(programId, requestPda(programId, user.publicKey, nonce)),
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .signers([providerOp])
        .rpc();

      const commitment = await program.account.responseCommitment.fetch(
        responsePda(programId, requestPda(programId, user.publicKey, nonce))
      );
      expect(Buffer.from(commitment.responseTxid)).to.deep.equal(responseTxid);
      expect(commitment.nTokens).to.equal(nTokens);
      expect(commitment.fingerprint.length).to.equal(64);
      expect(commitment.fingerprint[0]).to.be.closeTo(0.42, 1e-6);

      const req = await program.account.inferenceRequest.fetch(
        requestPda(programId, user.publicKey, nonce)
      );
      expect(req.status).to.deep.equal({ responded: {} });

      const p = await program.account.providerAccount.fetch(providerPda(programId, providerOp.publicKey));
      // EMA should have moved off zero after one measurement.
      expect(p.runningTokS).to.be.greaterThan(0);
    });

    it("submit_response from wrong provider is rejected", async () => {
      const n2 = new BN(2);
      // Set up a second request first
      await program.methods
        .submitRequest(n2, Array.from(randomTxid()), new BN(1_000_000), new BN(0))
        .accounts({
          user: user.publicKey,
          config: configPda(programId),
          model: modelPda(programId, MODEL_ID),
          queue: queuePda(programId, MODEL_ID),
          rotatingSeed: rotatingSeedPda(programId),
          request: requestPda(programId, user.publicKey, n2),
          stakeVault: vaultPda(programId),
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .signers([user])
        .rpc();

      // Assign it to real provider
      await program.methods
        .assignNext()
        .accounts({
          caller: providerOp.publicKey,
          queue: queuePda(programId, MODEL_ID),
          request: requestPda(programId, user.publicKey, n2),
          provider: providerPda(programId, providerOp.publicKey),
        })
        .signers([providerOp])
        .rpc();

      // Try to submit as a different "provider" (not registered)
      const imposter = Keypair.generate();
      await airdrop(provider.connection, imposter.publicKey, 10);
      try {
        await program.methods
          .submitResponse(Array.from(randomTxid()), makeFingerprint(), 50)
          .accounts({
            providerOperator: imposter.publicKey,
            config: configPda(programId),
            provider: providerPda(programId, imposter.publicKey),
            request: requestPda(programId, user.publicKey, n2),
            commitment: responsePda(programId, requestPda(programId, user.publicKey, n2)),
            systemProgram: anchor.web3.SystemProgram.programId,
          })
          .signers([imposter])
          .rpc();
        expect.fail("should have reverted");
      } catch (e: any) {
        expect(e.toString()).to.match(/AccountNotInitialized|constraint|NotAssignedProvider/);
      }
    });
  });

  // -----------------------------------------------------------------------
  //  Challenge path
  // -----------------------------------------------------------------------

  describe("challenge flow", () => {
    const nonce = new BN(10);
    const promptTxid = randomTxid();
    const responseTxid = randomTxid();

    before(async () => {
      // Set up: submit, assign, respond.
      await program.methods
        .submitRequest(nonce, Array.from(promptTxid), new BN(5_000_000), new BN(100_000))
        .accounts({
          user: user.publicKey,
          config: configPda(programId),
          model: modelPda(programId, MODEL_ID),
          queue: queuePda(programId, MODEL_ID),
          rotatingSeed: rotatingSeedPda(programId),
          request: requestPda(programId, user.publicKey, nonce),
          stakeVault: vaultPda(programId),
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .signers([user])
        .rpc();

      await program.methods
        .assignNext()
        .accounts({
          caller: providerOp.publicKey,
          queue: queuePda(programId, MODEL_ID),
          request: requestPda(programId, user.publicKey, nonce),
          provider: providerPda(programId, providerOp.publicKey),
        })
        .signers([providerOp])
        .rpc();

      await program.methods
        .submitResponse(Array.from(responseTxid), makeFingerprint(64, 0.1), 100)
        .accounts({
          providerOperator: providerOp.publicKey,
          config: configPda(programId),
          provider: providerPda(programId, providerOp.publicKey),
          request: requestPda(programId, user.publicKey, nonce),
          commitment: responsePda(programId, requestPda(programId, user.publicKey, nonce)),
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .signers([providerOp])
        .rpc();
    });

    it("challenge opens a Challenge PDA with the committee", async () => {
      const req = requestPda(programId, user.publicKey, nonce);
      await program.methods
        .challenge(new BN(50_000))
        .accounts({
          challenger: user.publicKey,
          config: configPda(programId),
          rotatingSeed: rotatingSeedPda(programId),
          request: req,
          commitment: responsePda(programId, req),
          challenge: challengePda(programId, req),
          stakeVault: vaultPda(programId),
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .remainingAccounts([
          { pubkey: verifier1.publicKey, isSigner: false, isWritable: false },
          { pubkey: verifier2.publicKey, isSigner: false, isWritable: false },
          { pubkey: verifier3.publicKey, isSigner: false, isWritable: false },
        ])
        .signers([user])
        .rpc();

      const ch = await program.account.challenge.fetch(challengePda(programId, req));
      expect(ch.verifierCommittee.length).to.equal(3);
      expect(ch.verdicts.length).to.equal(0);
      expect(ch.status).to.deep.equal({ open: {} });

      const r = await program.account.inferenceRequest.fetch(req);
      expect(r.status).to.deep.equal({ challenged: {} });
    });

    it("verifier_vote records guilty votes", async () => {
      const req = requestPda(programId, user.publicKey, nonce);
      for (const v of [verifier1, verifier2, verifier3]) {
        await program.methods
          .verifierVote(true, -5000) // scaled cosine
          .accounts({
            voter: v.publicKey,
            challenge: challengePda(programId, req),
          })
          .signers([v])
          .rpc();
      }
      const ch = await program.account.challenge.fetch(challengePda(programId, req));
      expect(ch.verdicts.length).to.equal(3);
      expect(ch.verdicts.every((v: any) => v.verdict === true)).to.be.true;
    });

    it("non-committee verifier cannot vote", async () => {
      const nonMember = Keypair.generate();
      await airdrop(provider.connection, nonMember.publicKey, 1);
      const req = requestPda(programId, user.publicKey, nonce);
      try {
        await program.methods
          .verifierVote(false, 9900)
          .accounts({
            voter: nonMember.publicKey,
            challenge: challengePda(programId, req),
          })
          .signers([nonMember])
          .rpc();
        expect.fail("should have reverted");
      } catch (e: any) {
        expect(e.toString()).to.match(/NotOnCommittee/);
      }
    });

    it("finalize_challenge slashes the provider on guilty quorum", async () => {
      const req = requestPda(programId, user.publicKey, nonce);
      const providerBefore = await program.account.providerAccount.fetch(
        providerPda(programId, providerOp.publicKey)
      );
      const userBalBefore = await balance(provider.connection, user.publicKey);

      await program.methods
        .finalizeChallenge()
        .accounts({
          caller: user.publicKey,
          config: configPda(programId),
          challenge: challengePda(programId, req),
          request: req,
          commitment: responsePda(programId, req),
          provider: providerPda(programId, providerOp.publicKey),
          providerWallet: providerOp.publicKey,
          challenger: user.publicKey,
          user: user.publicKey,
          stakeVault: vaultPda(programId),
        })
        .remainingAccounts([
          { pubkey: verifier1.publicKey, isSigner: false, isWritable: true },
          { pubkey: verifier2.publicKey, isSigner: false, isWritable: true },
          { pubkey: verifier3.publicKey, isSigner: false, isWritable: true },
        ])
        .signers([user])
        .rpc();

      const providerAfter = await program.account.providerAccount.fetch(
        providerPda(programId, providerOp.publicKey)
      );
      expect(providerAfter.stakeLamports.toNumber()).to.be.lessThan(
        providerBefore.stakeLamports.toNumber()
      );

      const r = await program.account.inferenceRequest.fetch(req);
      expect(r.status).to.deep.equal({ finalized: {} });

      // User should have received at least the refund + bond return + challenge reward.
      const userBalAfter = await balance(provider.connection, user.publicKey);
      expect(userBalAfter).to.be.greaterThan(userBalBefore);
    });
  });

  // -----------------------------------------------------------------------
  //  Housekeeping
  // -----------------------------------------------------------------------

  describe("housekeeping", () => {
    it("rotate_seed fails before window elapses", async () => {
      try {
        await program.methods
          .rotateSeed()
          .accounts({
            caller: user.publicKey,
            config: configPda(programId),
            rotatingSeed: rotatingSeedPda(programId),
          })
          .signers([user])
          .rpc();
        expect.fail("should have reverted");
      } catch (e: any) {
        expect(e.toString()).to.match(/SeedRotationNotReady/);
      }
    });
  });
});
