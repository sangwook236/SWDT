<html>
<body>

<?php

//include 'file.php?foo=1&bar=2';
//include 'http://www.example.com/file.php?foo=1&bar=2';

echo "A $color $fruit" . '<br>';  // A

include 'simple_inclusion.php';

echo "A $color $fruit" . '<br>';  // A green apple

say_hello();

if ((include 'simple_inclusion.php') == 'OK')
{
    echo 'Fail';
}

?>

</body>
</html>
