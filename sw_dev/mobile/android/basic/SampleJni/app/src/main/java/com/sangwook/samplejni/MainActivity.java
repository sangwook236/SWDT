package com.sangwook.samplejni;

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
        final SampleJni myJni = new SampleJni();
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //Toast.makeText(getApplicationContext(), "Hello, Android !!!", Toast.LENGTH_LONG).show();
                Toast.makeText(getApplicationContext(), myJni.getStringFromNative(), Toast.LENGTH_LONG).show();
            }
        });
    }
}
