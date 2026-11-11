# Commercial direction: reports for freelancers

## Confirmed decisions
Prem wants an app that could eventually charge users. The first audience is freelancers preparing client reports. The current spending preference is no spending: start with a demo. Prem can run and edit code with guidance. Uploading data must produce analysis automatically, without asking the user to write an explanation.

## Initial value proposition — a hypothesis to test
“Turn a client CSV into a readable first report, with visible calculations and data-quality warnings.”

The possible paid benefit is less time preparing and explaining reports. There is no evidence of demand, willingness to pay, customer acquisition cost, or profitability yet. Do not claim that the app will earn money simply because it is published.

## Phase 1: local demo, now
The current implementation produces a rule-based report without paid AI calls. It is deliberately not a complete AI product or commercial release. Test it with synthetic examples first. No deployment or purchase is needed.

For the next feedback round, ask a few freelancers to use non-sensitive sample exports. Watch whether they can upload a file, understand the findings, and produce something they would edit into a client deliverable. Ask what they had to rewrite, what was missing, and whether they would pay for a clearly scoped improvement. Do not ask them to upload confidential client files into an unreviewed prototype.

## Phase 2: a narrow paid pilot, only after useful feedback
Pick a more specific report type from actual user examples. For example, monthly marketing exports or project-delivery reports—but neither niche is selected yet.

Recommended initial billing experiment: a small report-credit package, not unlimited AI usage. Do not set a final price until costs per report, the useful output, and willingness to pay are understood. This is a proposed experiment, not a verified pricing recommendation.

Potential paid features to validate:
- Client-ready report layout with the freelancer's branding.
- Useful domain-specific calculations and a reviewable narrative.
- A reliable PDF export with appropriate charts.
- Repeatable report templates for the same client format.

## Components to design before accepting customer payments
- Customer sign-in and server-side authorization.
- A persistent customer/usage store, separate from temporary uploads.
- Hosted payment checkout from a provider supported in the seller's country; payment provider and seller country are not yet chosen.
- Signature-verified payment webhooks, replay/idempotency protection, and a credit ledger. Never grant access merely because the browser reaches a success URL.
- Server-side quota checks, atomic usage accounting, upload/concurrency limits, and provider spend caps.
- Private server-side API secrets if an LLM is enabled. Never expose API keys to the browser or source repository.
- A clear upload-retention policy, privacy notice, customer deletion path if anything is stored, and a review of data isolation and logging.
- Error handling, refund/support expectations, dependency maintenance, backup/recovery for billing records, and abuse monitoring.
- Honest descriptions of what is AI-generated, deterministic, uncertain, or unsupported.

This list is an engineering launch checklist, not legal or tax advice. The requirements vary by location and data processed.

## Costs and time
The original 2–3 hour target referred to a learning prototype. It does not cover a production-ready paid service. Estimate the paid pilot after selecting the report niche, hosting, authentication, and payment provider.

Local demo analysis has no AI usage fees. Hosting, payment fees, AI use, storage, and support may create costs after deployment. Pricing and profitability must be tested, not assumed.

## Role of the platforms
- **GitHub:** source control and project visibility. Publishing code does not create a payment system or customers. GitHub Pages should not host the paid SaaS under its published usage rules.[^1](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)
- **Hugging Face Spaces:** a possible app/demo host. Official documentation lists Gradio, Docker, and static Spaces. Current docs say compute-backed Gradio/Docker Spaces generally require a paid plan, with a stated limited ZeroGPU exception; recheck the account and plan before committing. Protected visibility keeps code private but the running app public. Customer payment gating must be implemented separately.[^2](https://huggingface.co/docs/hub/spaces-overview)

No hosting platform for the paid pilot has been selected. The supplied Dockerfile is preparation, not proof of a successful deployment or permission to process sensitive data.

## Decision to make after the first demo review
Choose whether to improve one report type, add an optional AI explanation using measured costs, or stop/pivot based on feedback. If feedback supports continuing, build the paid pilot in stages rather than adding every feature at once.

A later weekly product check-in could review feedback, test one sample dataset, and propose the next improvement. No routine is scheduled.
