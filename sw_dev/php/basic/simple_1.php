<html>
<body>

<?php
function sqr($par)
{ 
	return ($par * $par); 
} 

function inc(&$par)
{ 
	++$par; 
} 

$txt1 = "Hello World!";
$txt2 = "What a nice day!";
$txt = $txt1 . " " . $txt2;
echo $txt . "<br>" . strlen($txt) . '\n' . strpos($txt, "World") . '<br>';

echo sqr(5) . '<br>';
$par = 5;
inc($par);
echo $par;
?>

</body>
</html>
