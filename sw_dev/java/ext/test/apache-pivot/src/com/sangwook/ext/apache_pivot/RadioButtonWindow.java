package com.sangwook.ext.apache_pivot;

import java.net.URL;
 
import org.apache.pivot.beans.BXML;
import org.apache.pivot.beans.Bindable;
import org.apache.pivot.collections.Map;
import org.apache.pivot.util.Resources;
import org.apache.pivot.wtk.Alert;
import org.apache.pivot.wtk.Button;
import org.apache.pivot.wtk.ButtonGroup;
import org.apache.pivot.wtk.ButtonPressListener;
import org.apache.pivot.wtk.MessageType;
import org.apache.pivot.wtk.PushButton;
import org.apache.pivot.wtk.Window;

public class RadioButtonWindow extends Window implements Bindable
{
    @Override
    public void initialize(Map<String, Object> namespace, URL location, Resources resources)
    {
    	//selectButton_ = (PushButton)namespace.get("selectButton_");
        
        // Get a reference to the button group
        final ButtonGroup numbersGroup = (ButtonGroup)namespace.get("numbers");
 
        // Add a button press listener
        selectButton_.getButtonPressListeners().add(new ButtonPressListener() {
            @Override
            public void buttonPressed(Button button) {
                String message = "You selected \""
                    + numbersGroup.getSelection().getButtonData()
                    + "\".";
                Alert.alert(MessageType.INFO, message, RadioButtonWindow.this);
            }
        });
    }

    //private PushButton selectButton_ = null;
    @BXML private PushButton selectButton_ = null;
}
