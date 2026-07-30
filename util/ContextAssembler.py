
from datetime import timedelta
from util.context import Context, RetrievalContext

class ContextAssembler:
    """
    Builds the `messages` list to send to the API from a Context object.
    Handles chunk selection (dedup + recency), formatting, and injection.
    Does NOT mutate the Context or the caller's message list.
    """
 
    def __init__(
        self,
        max_chunks: int = 6,
        turn_window: int = 3,       # only consider chunks retrieved in last N turns
        max_age: timedelta | None = timedelta(minutes=30),
    ):
        self.max_chunks = max_chunks
        self.turn_window = turn_window
        self.max_age = max_age
 
    # -- public entry point --------------------------------------------------
 
    def build_messages(self, context: Context) -> list[dict]:
        messages = list(context.message_history)  # shallow copy, don't mutate original
 
        relevant = self._select_chunks(context)
        if relevant:
            injected_text = self._format_chunks(relevant)
            messages = self._inject(messages, injected_text)
 
        return messages
 
    # -- selection: dedup + recency filtering --------------------------------
 
    def _select_chunks(self, context: Context) -> list[RetrievalContext]:
        current_turn = context.current_turn_index()
        now = datetime.now()
 
        candidates = [
            c for c in context.retrieval_contexts
            if (current_turn - c.turn_index) <= self.turn_window
            and (self.max_age is None or (now - c.timestamp) <= self.max_age)
        ]
 
        # dedup by doc_id, keeping the highest-scoring instance of each doc
        best_by_doc: dict[str, RetrievalContext] = {}
        for c in candidates:
            existing = best_by_doc.get(c.doc_id)
            if existing is None or c.score > existing.score:
                best_by_doc[c.doc_id] = c
 
        deduped = sorted(best_by_doc.values(), key=lambda c: c.score, reverse=True)
        return deduped[: self.max_chunks]
 
    # -- formatting ------------------------------------------------------------
 
    def _format_chunks(self, chunks: list[RetrievalContext]) -> str:
        blocks = []
        for c in chunks:
            blocks.append(
                f'<retrieved_chunk doc_id="{c.doc_id}" collection="{c.collection}" '
                f'score="{c.score:.3f}">\n{c.text}\n</retrieved_chunk>'
            )
        return (
            "<retrieved_context>\n"
            + "\n".join(blocks)
            + "\n</retrieved_context>"
        )
 
    # -- injection: where in the message list it goes --------------------------
 
    def _inject(self, messages: list[dict], injected_text: str) -> list[dict]:
        """
        Injects retrieved context as its own user-role block appended right
        before the final user turn, so it reads as freshly-supplied context
        for the question just asked (rather than buried mid-history).
        """
        if not messages or messages[-1]["role"] != "user":
            # no trailing user turn to attach to; just append as its own turn
            messages.append({"role": "user", "content": injected_text})
            return messages
 
        last = messages[-1]
        combined_content = f"{injected_text}\n\n{last['content']}"
        new_last = {**last, "content": combined_content}
        return messages[:-1] + [new_last]
 
 
