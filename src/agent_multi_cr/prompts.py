from textwrap import dedent

from .auditors import MEMO_JSON_PREFIX


def _memo_json_footer(extra_append_clause: str) -> str:
    """
    Shared instructions for how reviewers should emit MEMO_JSON so that updates
    stay consistent across all prompt builders.
    """
    return f"""
    At the very end of your answer, on a separate line, add:

    {MEMO_JSON_PREFIX} {{"append": "...", "overwrite": false}}

    - `append` should contain any additional private notes you want to keep for yourself{extra_append_clause}
    - If `overwrite` is true, your existing memo will be replaced with `append`.
      Otherwise, `append` will be appended to your existing memo.
    """


def _shared_issue_format_guidelines() -> str:
    """
    Shared P0–P3 output format guidelines for reviewer prompts.
    """
    return """
    - Use valid Markdown.
    - Start with a one-sentence overall summary.
    - Then organize findings into four sections with these exact headings:
      - `P0 issues` – critical / blocking, must be fixed before merge.
      - `P1 issues` – important, should be fixed soon.
      - `P2 issues` – normal, worthwhile improvements.
      - `P3 issues` – minor / nice-to-have.
    - Under each section, use a numbered list.
      For each item include:
        - short title
        - approximate location (e.g., file or section, described in words is fine)
        - description of the problem
        - why it matters
        - a concrete suggestion for how to fix or improve it
        - if this item definitely needs a human to double-check, add
          `(NEEDS HUMAN REVIEW)` at the end of the item.
    """


def _reviewer_identity_block(reviewer_name: str) -> str:
    """
    Shared intro describing the reviewer and read-only workspace constraints.
    """
    return f"""
    You are **{reviewer_name}**, a senior code reviewer.

    You are working in a read-only copy of this repository in your current workspace.
    You **must not** modify real project files; your job is to analyze and comment only.
    """


def build_initial_review_prompt(
    reviewer_name: str,
    task_description: str,
    context_text: str,
    memo_text: str,
) -> str:
    """Prompt used for the very first independent review."""
    body = f"""
    {_reviewer_identity_block(reviewer_name)}

    The coordinator gives you this task:

    <REVIEW_TASK>
    {task_description}
    </REVIEW_TASK>

    The code under review lives in the files in your current working directory
    (a copy of the repository). You can and should use your tools (search,
    file inspection, git commands, etc.) to examine any code you need.

    The coordinator also provides this extra text context (may be empty, a note,
    a diff, or other text):

    ```text
    {context_text}
    ```

    Here is your current private memo for this repository. This memo is only for you and
    the coordinator; other reviewers and the final audience will not see it:

    <YOUR_PRIVATE_MEMO>
    {memo_text or "(empty)"}
    </YOUR_PRIVATE_MEMO>

    You may add new notes to this memo at the end of your answer.

    Your job:
    - Focus on the task above.
    - Identify potential bugs, missing edge cases, performance issues, security risks,
      and readability problems.
    - For each issue, explain clearly why it is a problem and how to fix or improve it.
    - If some parts are risky or ambiguous and require a human engineer, explicitly
      call that out.

    Output format (strict):
    {_shared_issue_format_guidelines()}
    """
    footer = _memo_json_footer(" (or an empty string if you have nothing to add).")
    return dedent(body + footer).strip()


def build_peer_review_prompt(
    reviewer_name: str,
    task_description: str,
    context_text: str,
    own_review: str,
    other_reviews_block: str,
    memo_text: str,
    round_index: int,
    max_rounds: int,
) -> str:
    """
    Prompt used when a reviewer cross-checks their own review against others.

    This is where reviewers reconcile disagreements among themselves before
    the arbiter aggregates the final result.
    """
    body = f"""
    {_reviewer_identity_block(reviewer_name)}

    The coordinator gave you this task:

    <REVIEW_TASK>
    {task_description}
    </REVIEW_TASK>

    The code under review lives in the files in your current working directory
    (a copy of the repository). You can and should use your tools (search,
    file inspection, git commands, etc.) to examine any code you need.

    The coordinator also provides this extra text context (may be empty, a note,
    a diff, or other text):

    ```text
    {context_text}
    ```

    This is cross-check round {round_index} of at most {max_rounds}.

    Here is your earlier review:

    <YOUR_INITIAL_REVIEW>
    {own_review}
    </YOUR_INITIAL_REVIEW>

    Here are the other reviewers' latest reviews:

    <OTHER_REVIEWS>
    {other_reviews_block or "(no other reviews yet)"}
    </OTHER_REVIEWS>

    Here is your private memo for this repository (only you and the coordinator can see this):

    <YOUR_PRIVATE_MEMO>
    {memo_text or "(empty)"}
    </YOUR_PRIVATE_MEMO>

    Your job in this round:
    - Compare your findings with the other reviewers.
    - For each high-priority (P0/P1) issue raised by anyone, decide whether you
      agree it is a real issue and whether the priority is appropriate.
    - If other reviewers identified valid issues that you missed, adopt them into
      your own list with an appropriate priority.
    - If you disagree with issues or priorities, say so clearly and explain why.
    - Then produce an updated, self-contained review that reflects your final.
      opinion, organized by P0–P3 just like your initial review.

    Output format (same as your initial review, but updated):
    {_shared_issue_format_guidelines()}
    - When you reference agreement or disagreement with others, do it briefly in
      the text (e.g., "Other reviewers agree that ...", "I disagree with ...").
    """
    footer = _memo_json_footer(
        " based on this cross-check (or an empty string if you have nothing to add)."
    )
    return dedent(body + footer).strip()


def build_followup_prompt(
    reviewer_name: str,
    task_description: str,
    context_text: str,
    initial_review: str,
    question: str,
    qa_snippet: str,
    memo_text: str,
) -> str:
    """Prompt used when the arbiter asks a reviewer a follow-up question."""
    body = f"""
    {_reviewer_identity_block(reviewer_name)}

    The coordinator originally gave you this task:

    <REVIEW_TASK>
    {task_description}
    </REVIEW_TASK>

    The code under review lives in the files in your current working directory
    (a copy of the repository). You can and should use your tools (search,
    file inspection, git commands, etc.) to examine any file you need.

    The coordinator also provides this extra text context (may be empty, a note,
    a diff, or other text):

    ```text
    {context_text}
    ```

    Here is your earlier review:

    <YOUR_INITIAL_REVIEW>
    {initial_review}
    </YOUR_INITIAL_REVIEW>

    Here is your private memo for this repository (only you and the coordinator can see this):

    <YOUR_PRIVATE_MEMO>
    {memo_text or "(empty)"}
    </YOUR_PRIVATE_MEMO>

    The review coordinator has a follow-up question for you about a specific part
    of this review:

    <QUESTION_FROM_COORDINATOR>
    {question}
    </QUESTION_FROM_COORDINATOR>

    If it helps, here is a short excerpt of your previous Q&A with the coordinator
    (you may ignore this if it does not seem useful):

    <YOUR_PREVIOUS_QA>
    {qa_snippet}
    </YOUR_PREVIOUS_QA>

    Please answer the coordinator's question carefully. Focus on clarifying:
    - whether you believe the discussed code is truly a problem that requires changes,
      or acceptable as-is;
    - how confident you are (0–1);
    - what you recommend in practice, in concrete terms;
    - when asked about a potential issue (even if you did not mention it before),
      explicitly say whether you **agree** it is a real issue, **disagree**, or are
      **uncertain**, and, if you think it is an issue, what approximate priority
      you would give it on the P0–P3 scale:
        - P0 – critical / blocking, must be fixed before merge.
        - P1 – important, should be fixed soon.
        - P2 – normal, worthwhile improvement.
        - P3 – minor / nice-to-have.

    Respond in **English** using Markdown.
    """
    footer = _memo_json_footer(
        " based on this Q&A (or an empty string if you have nothing to add)."
    )
    return dedent(body + footer).strip()


def build_arbiter_prompt(
    arbiter_name: str,
    task_description: str,
    context_text: str,
    auditors,
    initial_reviews,
    qa_history,
    max_queries: int,
    query_count: int,
    include_p2_p3: bool,
    allow_queries: bool,
) -> str:
    """Build the arbiter prompt."""
    reviewers_block_lines = [f"- {auditor.name}" for auditor in auditors]
    reviewers_block = "\n".join(reviewers_block_lines)

    initial_reviews_block_parts = [
        f'<REVIEW name="{name}">\n{review}\n</REVIEW>'
        for name, review in initial_reviews.items()
    ]
    initial_reviews_block = "\n\n".join(initial_reviews_block_parts)

    if qa_history:
        qa_lines = []
        for i, item in enumerate(qa_history, start=1):
            qa_lines.append(
                f"<QA_EXCHANGE index='{i}' reviewer='{item['reviewer']}'>\n"
                f"QUESTION:\n{item['question']}\n\n"
                f"ANSWER:\n{item['answer']}\n"
                f"</QA_EXCHANGE>"
            )
        qa_block = "\n\n".join(qa_lines)
    else:
        qa_block = "(no follow-up questions have been asked yet)"

    base_prompt = f"""
    You are the **arbiter code reviewer** named {arbiter_name}.

    The coordinator has defined the following review task:

    <REVIEW_TASK>
    {task_description}
    </REVIEW_TASK>

    You **do not** read or inspect the codebase directly. You only see what
    the auditors report and the Q&A between you and them. Your job is to
    reason about and reconcile those messages.

    The coordinator also provides the following extra text context
    (which may be empty, a short note, a diff, or other text):

    ```text
    {context_text}
    ```

    You coordinate several independent code reviewers (auditors). Each auditor only
    talks to you; they are not aware of each other, and you must **not** mention
    other auditors when you send them questions.

    You are given:
    1. The task description and shared context above.
    2. The list of auditors.
    3. Each auditor's initial review.
    4. A history of follow-up Q&A exchanges between you and individual auditors.

    Your job is to:
    - Understand where the auditors broadly agree.
    - Detect places where there seems to be uncertainty or disagreement.
    - Actively cross-check important issues across auditors, not just accept the
      first reviewer who mentioned them.
    - Ask *targeted* clarification questions to individual auditors when necessary.
    - Base all of your decisions on the auditors' written reviews and Q&A replies,
      not on your own reading of the code.
    - Eventually produce a single, unified review suitable to share with humans.

    Very important policy:
    - If, after a reasonable number of clarification questions, some part of the
      code is still ambiguous, risky, or controversial, you must mark it as
      **NEEDS HUMAN REVIEW** in your final output with a short explanation.
    - You should not try to "force" artificial consensus. It is OK to escalate
      genuinely unclear cases to humans.

    You have a hard limit of {max_queries} clarification questions in total.
    Up to now, you have already used {query_count} clarification questions.

    """

    if allow_queries:
        base_prompt += f"""

    You are called repeatedly in a loop. At each step, you must choose one of:
    1. Ask ONE auditor a clarification question about a specific aspect of the code.
    2. Produce the final unified review.

    How to use your clarification questions (very important):
    - Your goal is to **cross-check** the reviewers' findings and priorities.
    - Before you return a final review, you **must**:
      - for every high-priority (P0/P1) issue that any auditor proposes, ask
        each other auditor at least once whether they agree, disagree, or are
        uncertain about that issue, until you either get their stance or
        hit the {max_queries} limit;
      - make sure every auditor has been asked at least one direct question
        about the emerging list of P0/P1 issues.
    - When you ask about such an issue, describe the code and concern in neutral
      terms without mentioning other auditors by name, and ask the target auditor
      to state whether they agree, disagree, or are uncertain, and what priority
      they would assign.
    """
    else:
        base_prompt += """

    In this run you are called exactly once to produce the unified review.
    - You must *not* ask any clarification questions.
    - You must respond with "state": "final" and provide the best unified review
      you can based only on the existing reviews and Q&A.
    """

    base_prompt += f"""

    List of auditors (by name):
    {reviewers_block}

    Initial reviews from each auditor:

    {initial_reviews_block}

    Follow-up Q&A so far:

    {qa_block}

    === RESPONSE FORMAT (CRITICAL) ===

    You **must** respond with a single JSON object and nothing else.
    No Markdown code fences, no extra commentary, no trailing commas.

    There are only two valid shapes:

    1) To ask a new clarification question:

       {{
         "state": "query",
         "target_reviewer": "<exact auditor name from the list above>",
         "question": "<a short, clear question in English about one concrete issue in the context>",
         "reason": "<short explanation (for logs only) of why you need this question>"
       }}

    2) To finish and produce the unified review:

       {{
         "state": "final",
         "final_markdown": "<a full Markdown review to show to humans>"
       }}

    The final_markdown should:
    - Summarize overall impressions, in the context of the given task.
    - Present a unified, deduplicated issue list.
    - For **every issue that you decide to show**, assign a priority label in `P0`–`P3`
      and include it at the start of the item, e.g. `[P0] Title` or `[P2] Title`.
      - P0 = critical / blocking, must be fixed before merge.
      - P1 = important, should be fixed soon.
      - P2 = normal, worthwhile improvement.
      - P3 = minor / nice-to-have.
    - For each issue, indicate which auditors/models originally proposed it and
      which auditors appear to agree with it, using a short machine-readable
      line like:
        `Models: proposed_by=[...], agreed_by=[...]`
      where the names come from the auditor list above (for example
      `Codex[gpt-5.1|high]`, `Gemini[gemini-3-pro-preview]`).
      - `proposed_by` must list all auditors whose initial reviews clearly
        introduced or argued for the issue.
      - `agreed_by` should list auditors who either:
          * explicitly confirmed the issue in follow-up Q&A, or
          * clearly described the same underlying problem or risk in their own review.
      - Do **not** leave all `agreed_by` lists empty by default. Only leave
        `agreed_by` empty when you genuinely cannot infer any agreement from
        the reviews or Q&A.
    - When different auditors appear to disagree about whether an issue is real,
      or about its severity/priority, explicitly mention that disagreement in the
      item's text and still choose a single final P0–P3 priority based on the
      evidence you have. If that disagreement leaves residual uncertainty, mark
      the item with `NEEDS HUMAN REVIEW` and briefly explain why.
    - For any part that you believe is ambiguous, risky, or cannot be assessed
      confidently even after questions, explicitly mark it with
      `NEEDS HUMAN REVIEW` and briefly explain why.

    If you believe the information you already have is sufficient, you may
    choose "final" even if you have not used all {max_queries} questions.
    If you have already reached the limit of {max_queries} questions, you
    must choose "final" and make the best assessment you can, marking
    ambiguous parts as `NEEDS HUMAN REVIEW`.
    """

    if include_p2_p3:
        return dedent(base_prompt).strip()

    # Default mode: only show P0/P1 issues in detail.
    # P2/P3 issues may be considered internally but should not be listed
    # as full items in the final output.
    extra = """

    DISPLAY POLICY (when deciding what to show):
    - In the final_markdown, list only P0 and P1 issues in detail.
    - Do NOT include full items for P2 or P3 issues. You may briefly mention
      that lower-priority issues exist, but do not list them individually.
    """
    return dedent(base_prompt + extra).strip()
