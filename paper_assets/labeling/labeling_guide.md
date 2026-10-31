# Failure labeling guide

Each failed run in the sheet gets exactly one code from the list below. Choose the
code for the first decisive error: the earliest mistake without which the run could
have succeeded. Use a tool level code when one concrete action (or a missing action)
caused the failure. Use a MAST code when no single action is wrong by itself and the
failure lies in how the system is organised, how its agents communicate, or how it
checks its own work. Use OTHER only when nothing fits, and explain it in notes.

The auto_signals column lists automatic hints (for example wrong_final_state,
loop, handoff_missing_identifier). They are hints, not labels; check the trace
summary and the reply before deciding.

## How to label

* Each reviewer works only in their own copy: reviewer 1 in failures_to_label_reviewer1.csv
  (column label_1), reviewer 2 in failures_to_label_reviewer2.csv (column label_2).
* Do not open the other reviewer's copy and do not discuss individual runs until both
  copies are complete.
* Write the code exactly as given (for example WRONG_ARGUMENTS) and add a short reason in notes.
* When both copies are done, the coordinator runs `python -m analysis.agreement --tag <tag> --reviewer1 <file> --reviewer2 <file>`.

## MAST categories (Cemri et al., Why Do Multi-Agent LLM Systems Fail?, 2025)

### SYSTEM_DESIGN (system design issues)
The roles, prompts or structure of the system lead to the failure: an agent ignores its role
or the task specification, loses earlier context, or does not know when the task is done.
Example: the planner writes a plan that never includes the refund step the customer asked for.

### MISALIGNMENT (inter-agent misalignment)
Agents work against or past each other: information one agent has is not used by another,
the conversation drifts off the task, or one agent ignores another agent's report.
Example: the executor reports that the order is shipped, but the responder tells the customer it was cancelled.

### VERIFICATION (task verification)
The system ends without checking its result, or checks it wrongly, so a wrong or incomplete
outcome is reported as done.
Example: the agent states the refund went through although issue_refund returned an error.

## Tool level categories

### WRONG_TOOL
The agent calls a tool that cannot achieve the goal, or a different tool than the task needs.
Example: the customer asks for a refund and the agent calls cancel_order instead of issue_refund.

### WRONG_ARGUMENTS
The right tool is called but with a wrong or invented value (order ID, amount, method, department).
Example: issue_refund for 2920 taka by bkash is called with method cash_on_delivery.

### PREMATURE_STOP
The run ends before the needed actions are done: the agent replies, asks the customer for
confirmation, or stops after a read call without the write the task requires.
Example: the agent looks up the order and replies "please confirm" without cancelling it.

### LOOP
The agent repeats the same call or the same handoff without progress until it runs out of
calls or iterations.
Example: get_order with the same order ID is called five times and the budget is exhausted.

### POLICY_BREACH
The agent performs a write action the policy forbids, or skips a check the policy requires.
Example: a refund is issued nine days after delivery, outside the 7 day refund window.

### HANDOFF_ERROR
Information is lost or garbled when work is handed from one agent to another (supervisor to
specialist, planner to executor), or work goes to an agent that lacks the needed tool.
Example: the supervisor tells OrderAgent to cancel the order but leaves out the order ID.

### OTHER
Anything that fits none of the codes above, such as an infrastructure error or a scoring problem.
Example: the run stopped with an LLM timeout before any tool call.
