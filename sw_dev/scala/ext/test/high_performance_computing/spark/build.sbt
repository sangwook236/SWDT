name := "spark-example"

version := "1.0.0"

scalaVersion := "2.11.8"

// Additional libraries.
libraryDependencies ++= Seq(
	"org.apache.spark" %% "spark-core" % "2.3.1" % "provided"
)
