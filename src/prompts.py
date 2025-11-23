from textwrap import dedent


def build_initial_review_prompt(
    reviewer_name: str,
    task_description: str,
    context_text: str,
    memo_text: str,
) -> str:
    """Prompt used for the very first independent review."""
    return dedent(f"""
    You are a senior code reviewer acting as **{reviewer_name}**.

    You are working in your own private workspace. You **must not** modify the real project
    files. Treat the code under review as read-only; your job is to analyze and comment only.

    The coordinator gives you the following task:

    <REVIEW_TASK>
    {task_description}
    </REVIEW_TASK>

    Here is the shared code/context for this task (it may be a diff, a repository snapshot,
    a PR diff, or some other text representation of the project):

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
    1. Interpret the task description carefully and apply it to the given context.
       For example, if the task asks you to review the whole repo for security issues,
       focus on that; if it asks you to review a specific PR for correctness, focus on that.
    2. Identify potential bugs, missing edge cases, performance issues, security risks,
       and readability problems, relative to the task.
    3. For each issue, explain clearly why it is a problem.
    4. Suggest concrete, actionable fixes or improvements.
    5. If you believe some parts are risky or ambiguous and require a human engineer to
       inspect them, explicitly call that out.

    Output format (strictly follow this):
    - Use valid Markdown.
    - Start with a one-sentence overall summary.
    - Then organize findings by severity:
      - `Blocking issues`
      - `Important issues`
      - `Minor suggestions`
    - Under each severity, use a numbered list.
      For each item include:
        - A short title
        - Approximate location (e.g., file or section, described in words is fine)
        - Description of the problem
        - Why it matters
        - A concrete suggestion for how to fix or improve it
        - If you think this particular item *definitely* needs a human to double-check,
          add a tag like `(NEEDS HUMAN REVIEW)` at the end of the item.

    At the very end of your answer, on a separate line, add:

    MEMO_JSON: {{"append": "...", "overwrite": false}}

    - `append` should contain any additional private notes you want to keep for yourself
      (or an empty string if you have nothing to add).
    - If `overwrite` is true, your existing memo will be replaced with `append`.
      Otherwise, `append` will be appended to your existing memo.
    """).strip()


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
    return dedent(f"""
    You are a senior code reviewer acting as **{reviewer_name}**.

    You are working in your own private workspace. You **must not** modify the real project
    files. Treat the code under review as read-only; your job is to analyze and comment only.

    The coordinator originally gave you this task:

    <REVIEW_TASK>
    {task_description}
    </REVIEW_TASK>

    Here is the shared code/context for this task:

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
    - what you recommend in practice, in concrete terms.

    Respond in **English** using Markdown.

    At the very end of your answer, on a separate line, add:

    MEMO_JSON: {{"append": "...", "overwrite": false}}

    - `append` should contain any additional private notes you want to keep for yourself
      based on this Q&A (or an empty string if you have nothing to add).
    - If `overwrite` is true, your existing memo will be replaced with `append`.
      Otherwise, `append` will be appended to your existing memo.
    """).strip()


def build_arbiter_prompt(
    arbiter_name: str,
    task_description: str,
    context_text: str,
    auditors,
    initial_reviews,
    qa_history,
    max_queries: int,
    query_count: int,
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

    return dedent(f"""
    You are the **arbiter code reviewer** named {arbiter_name}.

    The coordinator has defined the following review task:

    <REVIEW_TASK>
    {task_description}
    </REVIEW_TASK>

    The shared code/context for this task is:

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
    - Ask *targeted* clarification questions to individual auditors when necessary.
    - Eventually produce a single, unified review suitable to share with humans.

    Very important policy:
    - If, after a reasonable number of clarification questions, some part of the
      code is still ambiguous, risky, or controversial, you must mark it as
      **NEEDS HUMAN REVIEW** in your final output with a short explanation.
    - You should not try to "force" artificial consensus. It is OK to escalate
      genuinely unclear cases to humans.

    You are called repeatedly in a loop. At each step, you must choose one of:
    1. Ask ONE auditor a clarification question about a specific aspect of the code.
    2. Produce the final unified review.

    You have a hard limit of {max_queries} clarification questions in total.
    Up to now, you have already used {query_count} clarification questions.

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
    - Present a unified, deduplicated issue list grouped by severity.
    - For each issue, mention whether it appears to be:
        - confidently agreed upon by the auditors, or
        - uncertain / controversial.
    - For any uncertain / controversial parts, explicitly label them with
      `NEEDS HUMAN REVIEW` and explain briefly why.

    If you believe the information you already have is sufficient, you may
    choose "final" even if you have not used all {max_queries} questions.
    If you have already reached the limit of {max_queries} questions, you
    must choose "final" and make the best assessment you can, marking
    ambiguous parts as `NEEDS HUMAN REVIEW`.
    """).strip()

