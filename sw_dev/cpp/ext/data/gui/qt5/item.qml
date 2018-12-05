import QtQuick 2.0

Item {
	id: item
	width: 100; height: 100

	property int someNumber: 100

	function myQmlFunction(msg) {
		console.log("Got message:", msg)
		return "some return value"
	}

	signal qmlSignal(string msg)
	signal qmlSignalObject(var obj)

	Rectangle {
		anchors.fill: parent
		objectName: "rect"
	}

	MouseArea {
		anchors.fill: parent
		onClicked: item.qmlSignal("Hello from QML")
		onDoubleClicked: item.qmlSignalObject(item)
	}
}
