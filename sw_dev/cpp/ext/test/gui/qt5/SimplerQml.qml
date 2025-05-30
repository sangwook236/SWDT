import QtQuick 2.6
import QtQuick.Window 2.2

Window {
	visible: true
	width: 640
	height: 480
	title: qsTr("Hello World")

	Rectangle {
		property alias mouseArea: mouseArea
		property alias textEdit: textEdit

		width: 360
		height: 360

		MouseArea {
			id: mouseArea
			anchors.fill: parent
		}

		TextEdit {
			id: textEdit
			text: qsTr("Enter some text...")
			verticalAlignment: Text.AlignVCenter
			anchors.top: parent.top
			anchors.horizontalCenter: parent.horizontalCenter
			anchors.topMargin: 20
			Rectangle {
				anchors.fill: parent
				anchors.margins: -10
				color: "transparent"
				border.width: 1
			}
		}

		anchors.fill: parent
		mouseArea.onClicked: {
			console.log(qsTr('Clicked on background. Text: "' + textEdit.text + '"'))
		}
	}
}
