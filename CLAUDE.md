# HireWise Claude Code Instructions

## Git authorship

- Use the Git `user.name` and `user.email` already configured for this repository.
- Never modify my Git identity.
- Never add Claude, Anthropic, Claude Code, an AI assistant, or another automated tool as an author, co-author, contributor, or attribution in commits.
- Never add `Co-Authored-By`, `Generated-By`, `Assisted-By`, or equivalent AI-attribution trailers.
- Commit messages should contain only the normal project commit subject/body.
- Never rewrite existing Git history, force-push, or modify remotes unless I explicitly request it.

## Existing project decisions

- Treat `config/generator_v1.yaml` as the frozen default experimental generator.
- Never modify the frozen generator merely to improve recommender results.
- A genuine generator change requires a separately versioned configuration and explicit documentation.
- Dataset 2 remains excluded from the modelling pipeline.
- Never expose generator-private variables to ordinary recommender models.
- Preserve the separation between model-visible data and generator-private/oracle truth.
- Respect temporal boundaries and prevent train/validation/test leakage.
- Use configuration rather than unexplained magic constants.
- Keep experiments deterministic using the project's established seeds/configuration.

## Development workflow

- Inspect existing implementation and documentation before modifying it.
- Do not silently change approved methodology.
- Run relevant tests before committing.
- Run appropriate independent verifier scripts when modifying their corresponding pipeline.
- Do not commit raw datasets, secrets, `.env` files, or inappropriate generated data.
- Keep commits focused and descriptive.
- Do not begin the next project day/milestone unless explicitly instructed.
- At the end of a milestone, provide a stopping report and wait for review.
