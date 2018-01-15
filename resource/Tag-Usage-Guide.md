## Documentation Tag

##### Hierarachy
- [-], [1]
	- =, (1), {1}
		- -, (i)

##### Status, Explanation or Constraint
- Use brace: {...}
- Usage
	- [-] {*~} 1 tab == 4 spaces.
		- = {/;need} change tag structure.
			- - {#;problem} our system malfunctions.
- Category
	- {status}: current status
		- important: *
		- completion: # or O
		- progress: ~ or -
		- delay: !
		- cancel: / or X

	- {need}: need or requirement.
	- {cause}: cause.
	- {solution}: solution.
	- {error}: error, bug, defect.
	- {problem}: problem or trouble.
	- {secondary-problem}: secondary/derived problem. <br/><br/>
	- {error-msg}: error message.
	- {warning-msg}: warning message. <br/><br/>
	- {cause}: cause.
	- {solution}: solution. <br/><br/>
	- {note}: note.

## Revision (History) Tag

##### Format
- When multiple lines
	- //--S [tag-id] yyyy/mm/dd: modifier-id
	- //	- {action-id}: <explanation>
	- //--E [tag-id] yyyy/mm/dd: modifier-id
- When single line & no explanation
	- //-- [tag-id] yyyy/mm/dd: modifier-id

##### Feature
- Can be nested
- tag-id
	- Tag-id need to be unique in the same file and can be omitted. But the bracket always has to exist.
	- In case that revision history exists, it's good that history-id is used.
	- When fixing bugs it seems good that error id is used as tag-id.
- modifier-id
	- Name or identifier of a subject to tag.
- action-id
	- Add some actions or explanation after the start tag.
	- identifier
		- {A}: add
		- {M}: modify
		- {R}: replace
		- {D}: delete


## Task Tag

##### To-Do Task
- Format
	- // TODO [task-id] >> `<brief description>`
	- //  - `<detailed explanation>` (optional)
- task-id
	- review
		- When an additional review is needed.
	- check
		- When more verification or tests are needed.
	- add, implement
		- When there are unimplements parts.

##### Fix Task
- Format
	- // FIXME [task-id] >> `<brief description>`
	- //  - `<detailed explanation>` (optional)
- task-id
	- modify, enhance, update, upgrade, fix, correct
		- When there are parts to be modified even though they are currently implemented.
	- delete
		- When some parts are written for other purposes like testing.
	- restore
		- When restoring parts which are implemented temporarily for testing into the original state.

##### Notice Task
- Format
	- // NOTICE [task-id] >> `<brief description>`
	- //  - `<detailed explanation>` (optional)
- task-id
	- caution
	- error, warn
	- info
		- e.g.)
			- NOTICE [info] {important} >>
	- note
		- REF [section] >> Note Taking
	- assumption, limitation

##### Reference Task
- Format
	- // REF [task-id] >> `<information on reference>`
	- //  - `<detailed explanation>` (optional)
- task-id
	- cross reference, cross ref, xref
		- {cross-reference-id, page no.}
			- cross-reference-id: Roman numerals, (i, ii, iii. iv, ...)
		- e.g.)
			- REF [cross reference] {iii: p.137} >>
			- REF [cross ref] {i: p.73, v: p.123} >>
			- REF [xref] {i: p.73, v: p.123} >>
	- site
	- file
	- paper
	- doc, presentation
	- video, audio

##### Pseudocode Programming Process (PPP) Task
- Format
	- // PPP [] {step:#} >> `<brief description>`
	- //  - `<detailed explanation>` (optional)

##### Design by Contract (DbC) Task
- Format
	- // CONTRACT [task-id] {required/optional} >> `<brief description>`
	- //  - `<detailed explanation>` (optional)
	- // DbC [task-id] {required/optional} >> `<brief description>`
	- //  - `<detailed explanation>` (optional)
	- // PRECONDITION [] {none/required/optional} >> `<brief description>`
	- //  - `<detailed explanation>` (optional)
	- // POSTCONDITION [] {none/required/optional} >> `<brief description>`
	- //  - `<detailed explanation>` (optional)
	- // INVARIANT [] {none/required/optional} >> `<brief description>`
	- //  - `<detailed explanation>` (optional)
- task-id
	- precondition
	- postcondition
	- invariant
- REF [file] >> ContractViolation and its derived classes in ${VIGRA_LIBRARY_HOME}/include/vigra/error.hxx

## Note Taking
- Use bullet
	- REF [site] >> http://bulletjournal.com/ <br/><br/>
	- .(dot): task
	- O(circle): event
	- -: note
	- *: important
	- X: complete
