package com.sangwook.simplejni;

import android.app.Activity;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import com.sangwook.externallib.Arithmetic;

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
        final ArithmeticJni arithmeticJni = new ArithmeticJni();  // Interface to native C++ classes in the same project and in an external native library.
        final Arithmetic arithmetic = new Arithmetic();  // Interface to Java class.
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //Toast.makeText(getApplicationContext(), "Hello, Android !!!", Toast.LENGTH_LONG).show();
                Toast.makeText(getApplicationContext(),
                        stringJni.getStringFromNative() +
                        "\n3 + 5 = " + arithmeticJni.add(3, 5) + ", 7 - 2 = " + arithmeticJni.sub(7, 2) +
                        "\n7 + 3 = " + arithmeticJni.add_in_lib(7, 3) + ", 2 - 5 = " + arithmeticJni.sub_in_lib(2, 5) +
                        "\n2 * 3 = " + arithmetic.mul(2, 3) + ", 6 / 5 = " + arithmetic.div(6, 5),
                        Toast.LENGTH_LONG).show();
            }
        });
    }
}
