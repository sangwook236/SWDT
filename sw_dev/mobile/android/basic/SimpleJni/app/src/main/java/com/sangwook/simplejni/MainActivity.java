package com.sangwook.simplejni;

import android.app.Activity;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

/**
 * Created by sangwook on 7/3/2017.
 */

public class MainActivity extends Activity {
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        Button button = (Button)findViewById(R.id.button);
        final StringJni stringJni = new StringJni();
        final ArithmeticJni arithmeticJni = new ArithmeticJni();
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //Toast.makeText(getApplicationContext(), "Hello, Android !!!", Toast.LENGTH_LONG).show();
                Toast.makeText(getApplicationContext(), stringJni.getStringFromNative() + "\n3 + 5 = " + arithmeticJni.add(3, 5) + ", 7 - 2 = " + arithmeticJni.sub(7, 2), Toast.LENGTH_LONG).show();
            }
        });
    }
}
