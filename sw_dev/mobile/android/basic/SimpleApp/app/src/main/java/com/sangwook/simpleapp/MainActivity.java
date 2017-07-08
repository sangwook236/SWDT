package com.sangwook.simpleapp;

import android.app.Activity;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import com.sangwook.externallib.Arithmetic;
import com.sangwook.externallib.TrigonometricJni;

public class MainActivity extends Activity {
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        Button button = (Button)findViewById(R.id.button);

        final StringJni stringJni = new StringJni();  // Interface to a native C++ class in the same project.
        final ArithmeticJni arithmeticJni = new ArithmeticJni();  // Interface to a native C++ class in the same project.
        final TrigonometricJni trigonometricJni = new TrigonometricJni();  // Interface to a native C++ class in an external native library.
        final Arithmetic arithmetic = new Arithmetic();  // Interface to a Java class in an external AAR.

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //Toast.makeText(getApplicationContext(), "Hello, Android !!!", Toast.LENGTH_LONG).show();
                Toast.makeText(getApplicationContext(),
                        stringJni.getStringFromNative() +
                        "\n3 + 5 = " + arithmeticJni.add(3, 5) + ", 7 - 2 = " + arithmeticJni.sub(7, 2) +
                        "\nsin(45) = " + trigonometricJni.sin(45.0 * Math.PI / 180.0) + ", cos(30) = " + trigonometricJni.cos(30.0 * Math.PI / 180.0) +
                        "\n2 * 3 = " + arithmetic.mul(2, 3) + ", 6 / 5 = " + arithmetic.div(6, 5),
                        Toast.LENGTH_LONG).show();
            }
        });
    }
}
