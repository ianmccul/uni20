# Uni20 Custom GPT GitHub Action Privacy Policy

- **Effective date:** 2026-07-19
- **Action domain:** `api.github.com`
- **Repository:** <https://github.com/Uni20-dev/uni20>

This policy covers the optional Custom GPT GitHub Actions described in
`github_repo_action.openapi.yaml` and `github_issue_action.openapi.yaml`. It is
intended for Uni20 maintainers and collaborators who configure a ChatGPT Custom
GPT to read or create limited content in the Uni20 GitHub repository.

## What the action does

The action calls GitHub's REST API for the `Uni20-dev/uni20` repository. It can:

- read public repository metadata, branches, commits, trees, blobs, file
  contents, code-search results, issues, and pull requests;
- create a GitHub issue only when the user explicitly requests issue creation or
  approves the final issue title and body.

The action does not operate a Uni20-controlled server or database between
ChatGPT and GitHub.

## Information sent to GitHub

When the action runs, ChatGPT may send GitHub the parameters needed for the
selected operation, such as:

- repository paths, commit refs, branch names, search queries, issue filters,
  and pagination parameters;
- for issue creation, the approved issue title, body, and optional labels.

If issue creation is enabled, the configured GitHub authentication token is sent
to GitHub by ChatGPT as part of the API request. Do not paste secrets, private
credentials, unpublished research notes, or personal data into issue text.

## Storage and visibility

The Uni20 project does not separately collect, log, or store action requests.
GitHub and OpenAI may process action requests according to their own terms and
privacy policies.

Content created through the issue-creation operation is stored by GitHub as a
GitHub issue. In a public repository, issue titles, bodies, labels, comments,
and metadata may be publicly visible.

## Authentication

Read-only repository operations should be configured without authentication.
Issue creation requires a GitHub token with the narrowest practical permissions
for `Uni20-dev/uni20`, normally repository metadata read access plus issues
read/write access.

## User control

Users should review action requests before allowing ChatGPT to run them. Issue
creation should be used only after the user has approved the final issue title
and body.

## Contact

For questions or corrections about this policy, open an issue or pull request in
the Uni20 repository.
