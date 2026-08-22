---
name: writing-rules
description: "Use when writing or reviewing any English prose in this repository: code comments, docstrings, commit messages, help and tooltip text, dialog and error messages, README/docs markdown, or release notes. Adapts ASD-STE100 (Simplified Technical English) into a concise rules table and a review checklist. Domain-neutral: applies to any technical base. Triggers: writing documentation, writing a docstring, writing GUI or CLI help text, reviewing text for clarity, technical writing style."
---

# Writing rules

This skill adapts ASD-STE100 (Simplified Technical English), the aerospace
controlled-language standard, for this repository. It governs every piece of
English prose: comments, docstrings, commit messages, help and tooltip text,
dialog and error messages, and every Markdown file under `docs/` or the repo
root.

ASD-STE100 was built so a technician can never misread an instruction. The
same discipline keeps documentation short, unambiguous, and easy to translate
or parse, by a human reader or by an agent. The rules are domain-neutral:
they apply to any technical base (engines, compilers, web services, data
pipelines, hardware).

Text is either **procedural** (steps, CLI usage, GUI instructions, a
tutorial) or **descriptive** (an explanation, a docstring, a module
overview, prose in `docs/`). The General rules apply to both. The
Procedural and Descriptive rules apply only to their own kind of text.

> SYNC NOTE: this file exists in two copies that must stay identical:
> `.agents/skills/writing-rules/SKILL.md` (canonical, any LLM) and
> `.claude/skills/writing-rules/SKILL.md` (Claude Code auto-discovery).
> If you edit one copy, apply the same edit to the other.

## General rules

| # | Do | Don't |
|---|----|----|
| G1 | Write in formal, technical English. "The parser did not find the closing tag." | Write in a casual or conversational tone. "Looks like the parser just couldn't find the tag." |
| G2 | Give each sentence one topic or one instruction. "Set the buffer size. Then run the test suite." | Chain unrelated ideas into one sentence. "Set the buffer size and run the test suite, which then writes the report." |
| G3 | Use active voice. "The scheduler dispatches each task." | Use passive voice when the actor is known. "Each task is dispatched by the scheduler." |
| G4 | Use only simple verb forms: infinitive, imperative, simple present, simple past, simple future, or past participle as an adjective. "The log file reached its size limit." | Build a compound tense with an auxiliary verb. "The log file has reached its size limit." |
| G5 | Use a past participle as an adjective to show a state. "Examine the corrupted sector." | Use it to build a passive construction. "The sector was corrupted by the failed write." |
| G6 | Write the full sentence: its verb, its articles, and any connector it needs. Don't use contractions. "If you use a custom configuration file, validate it first." | Write telegraphic text that drops the verb, the articles, or an adverbial connector to save space. "Custom config file: validate first." |
| G7 | Keep a multi-word noun to 3 words or fewer. "Connection retry limit." | Stack four or more nouns together. "Cache entry eviction policy check." |
| G8 | Use a single plain verb. "Remove the stale entry." | Build a phrasal verb from two words. "Take out the stale entry." |
| G9 | Use an "-ing" word only as a technical noun or as a modifier in one. "Open the Settings tab." | Use "-ing" as a verb form. "While debugging the crash, check the log." |
| G10 | Use the same term for the same thing every time. Always "access token," never switch to "auth string" mid-document. | Rotate synonyms for the same concept. "Access token" in one line, "auth string" in the next. |
| G11 | Use a verb to name an action. "Check the return value." | Turn the action into a noun. "Do a check of the return value." |
| G12 | Use an article before a noun. "Open the results tab." | Drop the article for a telegraphic style. "Open results tab." |
| G13 | Use plain, well-known words. "The deployment failed." | Use slang or jargon outside the project's own vocabulary. "The deploy went sideways." |
| G14 | Split related facts into separate sentences. "The value did not converge. Increase the iteration limit." | Join them with a semicolon. "The value did not converge; increase the iteration limit." |
| G15 | Use a plain sentence break instead of a dash. "The cache size changes the hit rate. This shifts the latency profile." | Use a dash to join two ideas. "The cache size changes the hit rate — this shifts the latency profile." |
| G16 | Add a connector when one sentence follows logically from the one before it: *however*, *therefore*, *thus*, *then*, *nevertheless*. "The table has no rows for this case. Therefore, the loader falls back to defaults." | Leave the logical link implicit. "The table has no rows for this case. The loader falls back to defaults." |
| G17 | Use "that" after verbs like "make sure," "confirm," or "show." "Make sure that the configuration file is loaded." | Drop "that" and risk a misread clause. "Make sure the configuration file is loaded." |
| G18 | Replace a pronoun with the noun it refers to, if more than one noun could fit. "If the pins are damaged, replace the pins." | Leave an ambiguous pronoun. "If the pins are damaged, replace them." |
| G19 | State what "this" refers to when more than one reading is possible. "If the tab is locked, this lock blocks editing." | Leave "this" to point at an unclear antecedent. "If the tab is locked, this blocks editing." |
| G20 | Spell out "for example," "that is," "and so on." | Use a Latin abbreviation. "e.g.," "i.e.," "etc." |
| G21 | Use a numbered or bulleted list for a sequence or a set of conditions with 3 or more items. | Bury a 3-step sequence inside one paragraph of prose. |
| G22 | Keep sentence length in check: about 20 words for an instruction, about 25 words for a description. | Write a long sentence with several clauses. |
| G23 | When an approved word does not fit, restructure the sentence around a word that does. "Make sure that you can see the oil level." | Force a word-for-word swap that reads as nonsense. "Make sure that the oil level is visible." (when "visible" is not an approved word here) |
| G24 | Re-read every sentence that uses "with." Confirm the reader cannot mistake it for association, means, or an instrument. "Seal the opening with sealant part number 4471." | Leave "with" open to more than one reading. "Install the panel with the fasteners." (unclear whether "with" means "using" or "together with") |
| G25 | In prose, write a sentence break instead of `--`, and write "approximately" instead of `~`. "The mesh is radial-azimuthal. Convergence is approximately quadratic." | Use `--` or `~` as a substitute for punctuation or a word in prose. "a radial--azimuthal mesh", "~quadratic convergence" |
| G26 | Write "and" or "or" between two words. "the input and output labels", "a timeout or retry target" | Join two words with a slash and leave the reader to guess which you mean. "Input/output labels", "a timeout/retry target" |
| G27 | Keep a parenthesis short, and put a fact the reader needs in its own sentence. | Hide a needed fact inside a long parenthesis. |
| G28 | Write a range in words. "5 to 30 retries" | Write a range with a hyphen. "5-30 retries" |
| G29 | Write a cross-reference in full. "Section 5" | Abbreviate a cross-reference. "Sec.5" |
| G30 | Say what the code or the theory computes. "the checksum of the payload", "queueing theory does not predict the tail latency here." | Give code or a theory human intent. "what the payload looks like to the hasher", "Queueing theory refuses to talk about tails." |
| G31 | Use American spelling. "behavior", "normalize", "labeled", "center" | Use British spelling. "behaviour", "normalise", "labelled", "centre" |
| G32 | State only what is factual and measurable, and keep a hedge exactly as strong as the source. "The solver converged in 12 iterations." "The run may have failed." | Use a superlative, a marketing word, an exaggeration or a non-technical vague word (robust, powerful, seamless, huge, the best). Stack hedges until the sentence claims nothing ("it may potentially help to improve"), or promote a hedge to a fact. |

**`G25`, `G26` and `G28` never apply to code.** A command-line flag keeps
its dashes (`--project`, `--set`, `--max-iter`). A hyphenated compound
adjective keeps its single hyphen (`a real-time system`, `a fixed-point
solver`). A file path, an operator, a numeric literal and a range inside a
code sample stay exactly as the code writes them. These rules govern
prose only.

All documentation in this project, including docstrings, helper text,
tooltips, and user-facing strings written in English, must follow this
table. Strings in other languages follow the project's language policy.

## Procedural rules (steps, CLI usage, GUI instructions)

| # | Do | Don't |
|---|----|----|
| P1 | Write one instruction per sentence, unless two actions happen at the same time. "Open the Geometry tab. Select the blade station." | Merge two sequential steps into one sentence. "Open the Geometry tab and select the blade station." |
| P2 | Write instructions in the imperative form. "Set the collective pitch." | Describe the instruction instead of giving it. "The collective pitch can be set." |
| P3 | State a condition first, then the command, separated by a comma. "If the airfoil table is missing, load a default polar." | Bury the condition after the command. "Load a default polar if the airfoil table is missing." |
| P4 | Use a note only to give information. "Note: the tip loss factor also affects the root region." | Put an instruction or a requirement inside a note. "Note: set the tip loss factor before running the case." |
| P5 | Name the concrete risk in a warning or an error message. "This deletes the project folder and its results." | State an abstract risk. "This action is not recommended." |

## Descriptive rules (explanations, docstrings, module overviews, prose)

| # | Do | Don't |
|---|----|----|
| D1 | Give information gradually, one subject per sentence. "BEMT couples blade element theory with momentum theory. It solves for the induced velocity at each station." | Front-load several facts into one dense sentence. "BEMT, which couples blade element and momentum theory, solves for induced velocity at each station using an iterative residual." |
| D2 | Open a paragraph with a topic sentence that states its subject. | Start a paragraph mid-detail, with the topic implied. |
| D3 | Keep one topic per paragraph, and keep each paragraph to 6 sentences or fewer. | Mix two topics, or let a paragraph run past 6 sentences. |

## How to review a text

1. Read the text once, for meaning only. Do not edit yet.
2. Decide whether it is procedural or descriptive, and pull in that
   section's rules (`P1`-`P5` or `D1`-`D3`) along with the General rules.
3. Go through every applicable rule, one at a time, from `G1` to the last
   rule in each table. For each one, check whether the text complies. This
   step is mandatory: do not skip straight to a general impression.
4. `G1`: confirm the tone is formal and technical, not casual or
   conversational.
5. `G2` and `P1`: split any sentence that holds more than one topic or
   instruction, unless the actions happen at the same time.
6. `G14`: split any sentence joined by a semicolon.
7. `G15`: replace any dash with a plain sentence break.
8. `G10`: look for synonym rotation, the same thing named two different
   ways. Pick one name and use it everywhere.
9. `G11`: look for a nominalization ("perform a check of"). Replace it
   with the verb ("check").
10. `G32`: delete every superlative, marketing word and exaggeration, and
    state the claim the text is hedging around. Do not change how strong
    the original claim was.
11. `G7` and `G8`: replace any remaining noun cluster over 3 words, or any
    phrasal verb, with a plain, specific one.
12. `G16`: add a connector where one sentence's meaning depends on the
    sentence before it.
13. `G23` and `G24`: confirm any reworded sentence keeps its original
    meaning, and that every "with" reads without ambiguity.
14. `G25`, `G26` and `G28`: in prose, replace `--`, `~`, a joining slash
    and a hyphenated range. First confirm the text is prose. Never touch a
    command-line flag, a compound adjective, a path, or a code sample.
15. `G20`, `G29`, `G30` and `G31`: replace a Latin abbreviation, an
    abbreviated cross-reference, a word that gives code human intent, and
    any British spelling.
16. `D3` (descriptive text only): confirm each paragraph has one topic and
    6 sentences or fewer.
17. Reread the whole text start to finish. Confirm every rule from step 3
    is satisfied, and that no fact or hedge was lost in the edit.

## Scope

This skill governs English prose and English user-facing strings. It does
not govern code identifiers, which follow the project's existing naming
conventions. It does not override a project rule about which language a
user-facing string uses; when such a string is written in English, this
skill applies to it. It complements, and does not override, any structural
rules for documentation files in the project's agent instructions.
