import javax.swing.*;

public class JOptionPaneTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		final String kind = JOptionPane.showInputDialog("동물의 종류를 입력하세요. (1:개, 2:물고기, 3:공룡, 0:종료)");
		String output;
		//if (kind.equals("1"))
		if (kind.equals("1"))
			output = "개";
		else if (kind.equals("2"))
			output = "물고기";
		else if (kind.equals("3"))
			output = "공룡";
		else
			output = "모르는 종";

		//JOptionPane.showMessageDialog(null, output);

		JTextArea showArea = new JTextArea();
	    showArea.setText(output);

		JOptionPane.showMessageDialog(null, showArea); 
	}

}
