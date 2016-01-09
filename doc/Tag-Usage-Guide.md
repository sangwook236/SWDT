## Documentation Tag

##### hierarachy
- [-]
- [1]
	- <->
	- <1>
		- ->
		- +>
		- :>
		- >>
		- <>
		- 1>
			- -.
			- 1.
			- i.

##### status, explanation or constraint
- use brace: {...}
- usage
	- [-] {*} 1 tab == 4 spaces.
	- -. {/,need} change tag structure.
	- -. {progress,problem} our system malfunctions.
- category
	- {current}: current status
		- completion, *
		- progress, - or ~
		- delay, !
		- cancel, /

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


## Revision (History) Tag

##### format
- when multiple lines
	- //--S [tag-id] yyyy/mm/dd: modifier-id
	- //	-. {action-id}: <explanation>
	- //--E [tag-id] yyyy/mm/dd: modifier-id
- when single line & no explanation
	- //-- [tag-id] yyyy/mm/dd: modifier-id

##### feature
- can be nested
- tag-id
	- tag id는 동일 file 내에서 unique하여야 하며 생략될 수 있다.
		- 그러나 bracket은 항상 존재하여야 한다.
	- revision history가 있을 경우 history id를 사용하는 것이 좋다.
	- bug 수정시에는 tag id를 error id로 하는 것이 좋을 듯하다.
- modifier-id
	- tagging을 한 사람의 name or identifier.
- action-id
	- start tag 후에 수행 동작과 설명을 추가할 수 있다.
	- identifier
		- {A}: add
		- {M}: modify
		- {R}: replace
		- {D}: delete


## Task Tag

##### format
- review
	- // TODO [review] >> <brief description>
	- //  -. <detailed explanation>
		- e.g.) 추가적인 검토가 필요한 경우.
- check
	- // TODO [check] >> <brief description>
	- //  -. <detailed explanation>
		- e.g.) 검증이나 테스트가 충분히 되지 않은 경우.
- add/implement
	- // TODO [add/implement] >> <brief description>
	- //  -. <detailed explanation>
		- 미구현 사항이 있는 부분에 사용.
- modify/enhance/update/upgrade/fix/correct
	- // FIXME [modify/enhance/update/upgrade/fix/correct] >> <brief description>
	- //  -. <detailed explanation>
		- 구현은 되어 있으나 (재사용성 등의 이유로) 수정이 필요한 부분에 사용.
- delete
	- // FIXME [delete] >> <brief description>
	- //  -. <detailed explanation>
		- e.g.) test를 위해 작성한 경우.
- restore
	- // FIXME [restore] >> <brief description>
	- //  -. <detailed explanation>
		- test 등을 위해 대체했던 부분을 원상태로 복귀.

- note
	- // NOTE [caution] >> <brief description>
	- //  -. <detailed explanation>

- reference
	- // REF [site/file/doc/paper] >> <brief description>
	- //  -. <reference site or document, etc.>
	- //  -. <additional comment>

- pseudocode programming process (PPP)
	- // PPP [] {step:#} >> <brief description>
	- //  -. <detailed explanation>

- design by contract (DbC)
	- // CONTRACT [precondition/postcondition/invariant] {required/optional} >> <brief description>
	- //  -. <detailed explanation>
	- //  [ref] ContractViolation and its derived classes in ${VIGRA_LIBRARY_HOME}/include/vigra/error.hxx
- precondition
	- // PRECONDITION [] {none/required/optional} >> <brief description>
	- //  -. <detailed explanation>
- postcondition
	- // POSTCONDITION [] {none/required/optional} >> <brief description>
	- //  -. <detailed explanation>
- invariant
	- // INVARIANT [] {none/required/optional} >> <brief description>
	- //  -. <detailed explanation>
