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
	- {secondary-problem}: secondary/derived problem.

	- {error-msg}: error message.
	- {warning-msg}: warning message.

	- {cause}: cause.
	- {solution}: solution.

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

##### Format
- Review
	- // TODO [review] >> `<brief description>`
	- //  - `<detailed explanation>`
		- e.g.) When an additional review is needed.
- Check
	- // TODO [check] >> `<brief description>`
	- //  - `<detailed explanation>`
		- e.g.) When more verification or tests are needed.
- Add / Implement
	- // TODO [add/implement] >> `<brief description>`
	- //  - `<detailed explanation>`
		- When there are unimplements parts.
- Modify / Enhance / Update / Upgrade / Fix / Correct
	- // FIXME [modify/enhance/update/upgrade/fix/correct] >> `<brief description>`
	- //  - `<detailed explanation>`
		- When there are parts to be modified even though they are currently implemented.
- Delete
	- // FIXME [delete] >> `<brief description>`
	- //  - `<detailed explanation>`
		- e.g.) When some parts are written for other purposes like testing.
- Restore
	- // FIXME [restore] >> `<brief description>`
	- //  - `<detailed explanation>`
		- When restoring parts which are implemented temporarily for testing into the original state.

- Note
	- // NOTE [caution] >> `<brief description>`
	- //  - `<detailed explanation>`

- Reference
	- // REF [site/file/doc/paper] >> `<brief description>`
	- //  - `<reference site or document, etc.>`
	- //  - `<additional comment>`

- Pseudocode Programming Process (PPP)
	- // PPP [] {step:#} >> `<brief description>`
	- //  - `<detailed explanation>`

- Design by Contract (DbC)
	- // CONTRACT [precondition/postcondition/invariant] {required/optional} >> `<brief description>`
	- //  - `<detailed explanation>`
		- REF [file] >> ContractViolation and its derived classes in ${VIGRA_LIBRARY_HOME}/include/vigra/error.hxx
- Precondition
	- // PRECONDITION [] {none/required/optional} >> `<brief description>`
	- //  - `<detailed explanation>`
- Postcondition
	- // POSTCONDITION [] {none/required/optional} >> `<brief description>`
	- //  - `<detailed explanation>`
- Invariant
	- // INVARIANT [] {none/required/optional} >> `<brief description>`
	- //  - `<detailed explanation>`

## Note Taking
- Use bullet
	- REF [site] >> http://bulletjournal.com/

	- .(dot): task
	- O(circle): event
	- -: note
	- *: important
	- X: complete
