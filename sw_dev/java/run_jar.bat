rem error
rem java -jar nonrunnable_jar.jar

java -jar runnable_jar.jar

java -classpath nonrunnable_jar.jar java_jar.Hello
java -classpath nonrunnable_jar.jar java_jar.Hi
java -classpath runnable_jar.jar java_jar.Hello
java -classpath runnable_jar.jar java_jar.Hi
