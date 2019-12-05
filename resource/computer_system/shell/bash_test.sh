#!/usr/bin/env bash

# REF [site] >>
#	http://johnstowers.co.nz/pages/bash-cheat-sheet.html
#	https://bash.cyberciti.biz/guide/Main_Page
#	http://tldp.org/HOWTO/Bash-Prog-Intro-HOWTO.html

# $0 - The name of the Bash script.
# $1 - $9 - The first 9 arguments to the Bash script. (As mentioned above.)
# $# - How many arguments were passed to the Bash script.
# $@ - All the arguments supplied to the Bash script.
# $? - The exit status of the most recently run process.
# $$ - The process ID of the current script.
# $USER - The username of the user running the script.
# $HOSTNAME - The hostname of the machine the script is running on.
# $SECONDS - The number of seconds since the script was started.
# $RANDOM - Returns a different random number each time is it referred to.
# $LINENO - Returns the current line number in the Bash script.

#------------------------------------------------------------

ARGS=$@
echo "args are $ARGS"

# Indexed array: Its keys are ordered integers
declare -a my_array
my_array=(foo bar)

my_array[0]=abc
my_array+=(baz)
my_array[5]=def
my_array+=(baz foobar)
unset my_array[1]  # Delete.

echo ${my_array[0]} ${my_array[1]}

echo ${my_array[@]}
echo ${my_array[*]}
for i in "${my_array[@]}"; do echo "$i"; done
for i in "${my_array[*]}"; do echo "$i"; done

echo "The array contains ${#my_array[@]} elements"  # Array size.

echo "Indexes: ${!my_array[@]}"  # Keys.
for index in "${!my_array[@]}"; do echo "$index"; done  # Keys.

unset my_array  # Delete an entire array.

ARRAY_VAR=(3 6 9)
for val in ${ARRAY_VAR[*]}
do
	echo "Value is ${val}".
done

# Associative array: Its keys are represented by strings.
declare -A my_array
my_array=([foo]=bar [baz]=foobar)

my_array[foo]="bar"
my_array+=([baz]=foobar [foobarbaz]=baz)
unset my_array[foo]  # Delete.

echo "The array contains ${#my_array[@]} elements"  # Array size.

echo "Keys: ${!my_array[@]}"  # Keys.
for key in "${!my_array[@]}"; do echo "$key"; done  # Keys.

unset my_array  # Delete an entire array.

# String.
STR=Korea
if [ "$STR" = "Korea" ]
then
	echo "Equal"
fi
if [ "$STR" != "KOREA" ]
then
	echo "Not equal"
fi

if [ -n $STR ]  # True if the string length is non-zero.
then
	echo "Non-zero length"
fi
if [ -z "" ]  # True if the string length is zero.
then
	echo "Zero length"
fi

#[ $# -eq 0 ] && { echo "Usage: $0 file1 file2 ... fileN"; exit 1; }
[ $# -eq 0 ] && { echo "Usage: $0 file1 file2 ... fileN"; }

#------------------------------------------------------------

a=`ls -l`
echo $a
echo "$a"  # The quoted variable preserves whitespace.

arch=$(uname -m)
echo "$arch"

for i in $(seq 0 1 31)
do
    #mkdir 10.$(printf %02d $i).2019
    echo "10.$(printf %02d $i).2019"
done

# The shell tries to execute 'echo' and 'VAR=value' as two separate commands.
$(echo VAR=value)
echo $VAR

# The shell merges (concatenates) the two strings 'echo' and 'VAR=value', parses this single unit according to appropriate rules and executes it.
eval $(echo VAR=value)
echo $VAR

a="ls | more"
$a
eval $a

#------------------------------------------------------------
# If.

if true
then
	echo "True"
else
	echo "False"
fi

VAR=2

if [ $VAR -eq 1 ]
then
	echo "VAR is 1".
elif [ $VAR -eq 2 ]
then
	echo "VAR is 2".
else
	echo "VAR is greater than 2".
fi

#------------------------------------------------------------
# For.

for id in 2 4 6 8 10
do
	echo "ID is ${id}".
done

for id in {1..5}
do
	echo "ID is ${id}".
done

for (( id=1; id<=5; ++id ))
do
	echo "ID is ${id}".
done

files="/etc/passwd /etc/group /etc/shadow /etc/gshdow"
for f in $files
do
	[ -f $f ] && echo "$f file found." || echo "$f file not found."
done

for f in $(ls /tmp/*)
do
	echo $f
done

for id in `seq 1 5`
do
	echo $id
done

#------------------------------------------------------------
# Pattern matching.

a=something
if [[ $a == +(some|any)thing ]]; then echo yes; else echo no; fi

a=nothing
if [[ $a == +(some|any)thing ]]; then echo yes; else echo no; fi

#------------------------------------------------------------
# Regular expression.

# . or Dot will match any character
# [ ] will match a range of characters
# [^ ] will match all character except for the one mentioned in braces
# * will match zero or more of the preceding items
# + will match one or more of the preceding items
# ? will match zero or one of the preceding items
# {n} will match 'n' numbers of preceding items
# {n,} will match 'n' number of or more of preceding items
# {n m} will match between 'n' & 'm' number of items
# {,m} will match less than or equal to m number of items
# \ is an escape character, used when we need to include one of the metcharacters is our search.

# [[ "string" =~ pattern ]] performs a regular expression match.

digit=9
if [[ $digit =~ [0-9] ]]; then
    echo "$digit is a digit"
else
    echo "OOPS"
fi

INT=-5
if [[ "$INT" =~ ^-?[0-9]+$ ]]; then
	echo "INT is an integer."
else
	echo "INT is not an integer." >&2
	exit 1
fi

#------------------------------------------------------------
# Function.

function quit {
	exit
}  
function greet {
	echo "Hello $1"
}  
greet World
quit
