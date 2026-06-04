import logging
import sys
from pathlib import Path
from typing import TypedDict, Annotated, Sequence

# Allow running this file directly (e.g. via IDE "Run") in addition to
# `python -m RAG.RAG_graph`. Both need the repo root on sys.path so that
# `from RAG.rag import ...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError
from dotenv import load_dotenv

from RAG.rag import CLOIndentureRAG

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("RAG.rag").setLevel(logging.INFO)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Resolve the DB path from this file's location, not the process working
# directory: running RAG_graph.py from the IDE sets cwd to the RAG/ folder, so
# the default relative "RAG/chroma_db" would resolve to RAG/RAG/chroma_db (an
# empty DB) and every search would report "no indentures ingested".
_HERE = Path(__file__).resolve().parent  # .../RAG
rag = CLOIndentureRAG(persist_dir=str(_HERE / "chroma_db"))


@tool
def list_indentures() -> str:
    """List all CLO indenture documents (deals) currently available in the
    RAG system. Returns each indenture's `doc_id` (a human-readable
    identifier you must pass to `search_clo_indenture` to scope a search
    to a single deal) along with basic stats.

    Call this tool at the START of a conversation if the user has not
    yet identified which deal they want to ask about, OR whenever you
    are unsure which `doc_id` to use. Each session should resolve the
    deal once, then reuse the same `doc_id` for every subsequent
    `search_clo_indenture` call.
    """
    docs = rag.list_documents()
    if not docs:
        return "No indentures have been ingested yet."
    lines = ["Available indentures (use `doc_id` to scope searches):"]
    for d in docs:
        lines.append(
            f"  - doc_id={d['doc_id']!r}  "
            f"({d['n_chunks']} chunks, {len(d['sections'])} sections)"
        )
    return "\n".join(lines)


@tool
def search_clo_indenture(doc_id: str, query: str) -> str:
    """Search a specific CLO (Collateralized Loan Obligation) indenture
    document using semantic retrieval and return the top relevant chunks.

    Use this tool whenever the user asks about CLO indenture terms,
    definitions, covenants, tests, calculations, or any specifics that
    require evidence from the indenture document (e.g., CCC/Caa excess,
    overcollateralization test, interest coverage test, eligibility
    criteria, waterfall, concentration limits, etc.).

    IMPORTANT — you MUST know the `doc_id` before calling this tool. If
    you don't yet know which deal the user is asking about, call
    `list_indentures` first (or ask the user). Never guess a `doc_id`.

    IMPORTANT — one topic per call. Each query must target a single,
    coherent topic. If the user asks about multiple unrelated topics in
    one turn (e.g. "what's the closing date and who is the trustee?"),
    issue one tool call per topic in the same turn (the framework will
    run them in parallel). Do NOT combine unrelated topics into a single
    query — mixing topics dilutes the embedding and degrades retrieval.

    Args:
        doc_id: The exact `doc_id` of the indenture to search, as
            returned by `list_indentures`. Required — searches without a
            doc_id would mix evidence across deals and produce wrong
            answers (defined terms like "Closing Date" or "Trustee" are
            deal-specific).
        query: A focused natural-language question or phrase describing
            exactly what to look up in the indenture, covering ONE topic
            only. Rewrite the user's request into a self-contained query —
            include the specific term, defined concept, or calculation
            being asked about, using prior conversation context where
            needed (e.g. if the user previously asked about "CCC/Caa
            excess" and now asks "how to calculate the excess adjustment",
            pass "CCC/Caa excess adjustment calculation" as the query).

    Returns:
        A formatted string containing the top retrieved chunks, each
        with its rank, section metadata, and text body.
    """
    results = rag.retrieve(
        query=query,
        top_k=10,
        use_rerank=False,
        max_definition_chunks=6,
        doc_id=doc_id,
    )
    if not results:
        return "No relevant chunks found in the CLO indenture for that query."

    parts: list[str] = []
    for r in results:
        meta = r.get("metadata", {}) or {}
        section_id = meta.get("section_id", "")
        section_title = meta.get("section_title", "")
        page_start = meta.get("page_start", "")
        page_end = meta.get("page_end", "")
        pages = (
            f"{page_start}-{page_end}" if page_start != page_end else f"{page_start}"
        )
        header = (
            f"[Rank {r['rank']}] section_id={section_id} | "
            f"title={section_title} | pages={pages} | "
            f"score={r['score']:.4f}"
        )
        parts.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(parts)


tools = [list_indentures, search_clo_indenture]
# gemini-3.5-flash (not flash-lite): the lite tier is too weak to reliably
# (a) emit multiple parallel tool calls in one turn and (b) keep iterating in
# the agentic loop instead of wrapping up early. temperature is kept low so
# the model follows the "one topic per call" / "don't defer to the user"
# instructions consistently rather than wandering.
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", temperature=0.1
).bind_tools(tools)


def llm_node(state: State) -> State:
    system_prompt = (
        "You are an expert assistant on CLO (Collateralized Loan Obligation) "
        "indentures. You have access to two tools: "
        "(1) `list_indentures` — returns the available indenture documents "
        "and their `doc_id`s; "
        "(2) `search_clo_indenture(doc_id, query)` — retrieves the top "
        "relevant chunks from ONE indenture via RAG. "
        ""
        "ESTABLISH THE DEAL FIRST. Every CLO question is scoped to a "
        "specific deal, so you MUST know which `doc_id` to use before "
        "calling `search_clo_indenture`. On the very first user turn of "
        "a session (or any time the deal is unclear), call "
        "`list_indentures` to see what is available, then match it to "
        "the user's stated deal name. Call `list_indentures` at most "
        "ONCE for this — never call it twice in a row. If, after one "
        "listing, the user's message does not clearly identify one of "
        "the listed `doc_id`s, STOP and ask the user to pick one from "
        "the list; do NOT guess a `doc_id`, and do NOT re-list in a "
        "loop. Once the deal is established, REUSE the same `doc_id` "
        "for every subsequent `search_clo_indenture` call in the "
        "conversation without re-listing. "
        ""
        "Decide for each user turn whether you need to call "
        "`search_clo_indenture`: call it when the user asks about a "
        "specific indenture term, definition, test, or calculation that "
        "you need evidence for. Reuse information from earlier "
        "ToolMessages when the user's follow-up can be answered from "
        "what was already retrieved — do not call the tool redundantly. "
        "When you do call it, rewrite the user's request into a "
        "self-contained query that incorporates context from earlier "
        "messages (e.g. resolve pronouns and implicit references). "
        ""
        "ONE TOPIC PER TOOL CALL. If the user asks about multiple "
        "unrelated topics in one turn, emit one `search_clo_indenture` "
        "call per topic IN THE SAME TURN — they run in parallel. Never "
        "combine unrelated topics into a single query, because mixing "
        "topics dilutes the embedding vector and causes the retriever "
        "to return chunks that are mediocre matches for both rather "
        "than strong matches for either. Only combine terms in one "
        "query when they are part of the same concept (e.g. 'CCC/Caa "
        "excess adjustment calculation'). Before searching, silently "
        "decompose the user's turn into its distinct sub-questions, then "
        "fire one focused call for each. WORKED EXAMPLE — the request "
        "'for the initial term loans, get the coupon rate (benchmark + "
        "spread), maturity date, and any payment schedule' is THREE "
        "distinct topics and must become THREE parallel calls: "
        "(1) 'Applicable Rate / Applicable Margin spread over SOFR for "
        "Initial Term Loans'; (2) 'Initial Term Loan Maturity Date "
        "definition'; (3) 'Initial Term Loan amortization repayment "
        "schedule'. Do not collapse them into one query. "
        ""
        "KEEP GOING UNTIL THE QUESTION IS ANSWERED — you are an "
        "autonomous agent, not a one-shot lookup. After each batch of "
        "tool results, judge whether you now have the CONCRETE values "
        "the user asked for. If a result only gives you a pointer "
        "instead of the value — e.g. a defined term whose body is "
        "'has the meaning specified in Section 2.01(a)', or a rate that "
        "is expressed as 'Applicable Rate' / 'Applicable Margin' — that "
        "is NOT an answer; immediately issue a FOLLOW-UP "
        "`search_clo_indenture` call that chases the cross-reference "
        "(search the referenced section, or the 'Applicable Rate' / "
        "'Applicable Margin' definition and any pricing grid) to "
        "retrieve the actual number or date. Loop like this — search, "
        "read, search again — for as many rounds as it takes. "
        ""
        "NEVER defer RETRIEVAL work back to the user once the deal is "
        "known. Do NOT end your turn with offers like 'let me know if "
        "you want me to search for X' or 'I would need to look up Y' — "
        "if you can see the next search to run, just run it. Stop and "
        "produce a final answer (plain text, no tool call) ONLY when: "
        "(a) you have the concrete values the user asked for; (b) you "
        "have genuinely exhausted reasonable reformulations and the "
        "document does not contain them — state plainly what is and "
        "isn't in the document, citing what you did find; or (c) you "
        "cannot tell which deal/`doc_id` to search even after listing "
        "once — in that case ask the user to pick from the available "
        "`doc_id`s. That deal-selection question is the ONLY acceptable "
        "hand-back to the user; never loop on `list_indentures` instead "
        "of simply asking it. "
        ""
        "Cite the section_id from the retrieved chunks in your final "
        "answer whenever possible."
    )
    response = llm.invoke([SystemMessage(system_prompt)] + list(state["messages"]))
    return {"messages": [response]}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "continue"
    return "end"


graph = StateGraph(State)
graph.add_node("llm", llm_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {"continue": "tools", "end": END},
)
graph.add_edge("tools", "llm")

app = graph.compile()

# Safeguard for the now-more-persistent agentic loop. Each round trip
# (llm step + tools step) consumes 2 toward this limit, so 16 allows ~8
# search→read→search iterations before LangGraph raises GraphRecursionError.
# Enough for chasing cross-references; low enough to stop a runaway loop.
RECURSION_LIMIT = 16


_WIDTH = 80


def _hr(char: str = "-") -> None:
    print(char * _WIDTH)


def _extract_text(content: object) -> str:
    """Return readable text from a LangChain message content payload.

    Gemini (and some other providers) can return `content` as a list of
    content-part dicts (e.g. [{"type": "text", "text": "..."}]) instead of
    a plain string. Flatten that down to just the human-readable text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
        return "".join(parts)
    return str(content)


def _print_ai(msg: BaseMessage) -> None:
    print()
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            _hr("-")
            print(f"AI -> tool call: {tc['name']}")
            for k, v in tc["args"].items():
                print(f"    {k}: {v!r}")
            _hr("-")
    else:
        _hr("=")
        print("AI (final answer):")
        _hr("=")
        print(_extract_text(msg.content))
        _hr("=")


def _print_tool(msg: BaseMessage, preview_chars: int = 240) -> None:
    print()
    _hr("-")
    print(f"Tool result: {getattr(msg, 'name', '<tool>')}")
    _hr("-")
    chunks = (msg.content or "").split("\n\n---\n\n")
    for chunk in chunks:
        header, _, body = chunk.partition("\n")
        body = body.strip().replace("\n", " ")
        preview = body[:preview_chars] + ("..." if len(body) > preview_chars else "")
        print()
        print(f"  {header}")
        print(f"    {preview}")
    _hr("-")


def _print_new_message(msg: BaseMessage) -> None:
    msg_type = getattr(msg, "type", "")
    if msg_type == "ai":
        _print_ai(msg)
    elif msg_type == "tool":
        _print_tool(msg)


if __name__ == "__main__":
    messages: list[BaseMessage] = []
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        messages.append(HumanMessage(content=user_input))
        try:
            for update in app.stream(
                {"messages": messages},
                stream_mode="updates",
                config={"recursion_limit": RECURSION_LIMIT},
            ):
                for _node, payload in update.items():
                    for m in payload.get("messages", []):
                        messages.append(m)
                        _print_new_message(m)
        except GraphRecursionError:
            print(
                f"\n[stopped after {RECURSION_LIMIT} steps — the agent kept "
                "searching without converging. Try a more specific question.]"
            )
