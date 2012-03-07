#include "../HelloJniClass.h"
#include <jni.h>
#include <iostream>


#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jstring JNICALL Java_JNI_1Message_Message(JNIEnv *env, jobject jobj, jstring input)
{
	jboolean iscopy;
	static char outputbuf[20];

	const char *buf = env->GetStringUTFChars(input, &iscopy);  // 입력 String 읽어오는 함수
	std::cout << "\nDLL receive Data from JAVA : " << buf << std::endl;   // 입력받은 내용을 출력
	strcpy(outputbuf, "Delicious !!\n"); 
	const jstring js = env->NewStringUTF(outputbuf);  // 출력할 내용의 java버퍼에 output버퍼값을 셋팅

	return js; // java버퍼 리턴
}

#ifdef __cplusplus
}
#endif