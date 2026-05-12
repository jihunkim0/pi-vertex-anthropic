/**
 * Verification script for GCP Authentication.
 * 
 * Simulates the exact fetch request the extension makes to Vertex AI,
 * verifying that the `X-Goog-User-Project` header and user credentials 
 * correctly resolve 403 PERMISSION_DENIED errors.
 * 
 * Usage:
 *   npx tsx scripts/verify-auth.ts
 */
import { __test__ } from "../index.ts";
import { execFile } from "child_process";
import { promisify } from "util";
import fs from "fs";
import path from "path";

const execFileAsync = promisify(execFile);
const { GcpAuthClient } = __test__;

const defaultExecutor = {
	execute: async (command: string, args: string[]) => {
		const { stdout } = await execFileAsync(command, args);
		return stdout;
	}
};

async function run() {
	console.log("🔍 Starting GCP Authentication Verification...");
	
	// Read persisted config
	const authPath = path.join(process.env.HOME || "", ".pi/agent/auth.json");
	if (!fs.existsSync(authPath)) {
		console.error("❌ auth.json not found. Run `/login` in Pi first.");
		process.exit(1);
	}

	const data = JSON.parse(fs.readFileSync(authPath, "utf-8"));
	const cred = data["vertex-anthropic"];
	
	if (!cred || cred.type !== "oauth") {
		console.error("❌ Valid vertex-anthropic oauth credentials not found in auth.json.");
		process.exit(1);
	}

	const project = cred.project;
	const account = cred.account;
	const region = cred.region || "us-east5";

	console.log(`📌 Target Project: ${project}`);
	console.log(`📌 Target Account: ${account || "Default"}`);
	console.log(`📌 Target Region:  ${region}`);
	
	const gcpAuthClient = new GcpAuthClient(defaultExecutor);
	
	const body = {
		anthropic_version: "vertex-2023-10-16",
		messages: [{ role: "user", content: "hello from verify-auth script" }],
		max_tokens: 10
	};
	
	const url = `https://${region}-aiplatform.googleapis.com/v1/projects/${project}/locations/${region}/publishers/anthropic/models/${process.env.VERTEX_MODEL || "claude-3-5-sonnet-v2.0@20241022"}:streamRawPredict`;
	
	console.log(`\n🚀 Fetching token and making request to Vertex AI...`);
	const response = await gcpAuthClient.fetchWithAuth(
		url,
		{
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				"X-Goog-User-Project": project,
			},
			body: JSON.stringify(body),
		},
		"gcloud",
		account,
		{ maxRetries: 1 }
	);
	
	console.log(`\n📡 Status: ${response.status} ${response.statusText}`);
	
	if (response.ok) {
		console.log("✅ Authentication successful! The `X-Goog-User-Project` header resolved quota billing correctly.");
	} else {
		console.error("❌ Request failed.");
		const text = await response.text();
		console.error(text);
	}
}

run().catch(console.error);
