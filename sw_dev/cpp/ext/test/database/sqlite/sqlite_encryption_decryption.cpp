//#include "stdafx.h"
#if defined(WIN32)
#include <windows.h>
//#include <wincrypt.h>
#endif
#if defined(__linux) || defined(__linux__) || defined(linux) || defined(__unix) || defined(__unix__) || defined(unix)
#include <sqlite3.h>
#else
#include <sqlite/sqlite3.h>
#endif
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

#define ENCRYPT_ALGORITHM   CALG_RC4
#define KEYLENGTH           0x00800000

void encrypt(sqlite3_context *context, int argc, sqlite3_value **argv)
{
#if defined(WIN32)
	if (2 != argc)
	{
		sqlite3_result_error(context, "error: Invalid param count", -1);
		return;
	}
	else
	{
		const std::string rawData((const char *)sqlite3_value_blob(argv[0]));
		const std::string keyPhrase((const char *)sqlite3_value_text(argv[1]));

		if (!keyPhrase.empty())
		{
			HCRYPTPROV hCryptProv = NULL;

			// get the Handle to the default provider
			if (!CryptAcquireContext(&hCryptProv, NULL, MS_ENHANCED_PROV, PROV_RSA_FULL, 0))
			{
				if (GetLastError() == NTE_BAD_KEYSET)
				{
					if (!CryptAcquireContext(&hCryptProv, NULL, MS_DEF_PROV, PROV_RSA_FULL, CRYPT_NEWKEYSET))
					{
						// CryptAcquireContext() failed.
						sqlite3_result_error(context, "error: CryptAcquireContext()", -1);
						return;
					}
				}
				else
				{
					// CryptAcquireContext() failed.
					sqlite3_result_error(context, "error: CryptAcquireContext()", -1);
					return;
				}
			}

			HCRYPTHASH hHash = NULL;

			// create Hash object
			if (!CryptCreateHash(hCryptProv, CALG_MD5, 0, 0, &hHash) || !hHash)
			{
				// error during CryptCreateHash()
				sqlite3_result_error(context, "error: CryptCreateHash()", -1);

				CryptReleaseContext(hCryptProv, 0);
				return;
			}

			// hash the password
			if (!CryptHashData(hHash, (const unsigned char *)keyPhrase.c_str(), keyPhrase.length(), 0))
			{
				// error during CryptHashData
				sqlite3_result_error(context, "error: CryptHashData()", -1);

				CryptDestroyHash(hHash);
				CryptReleaseContext(hCryptProv, 0);
				return;
			}

			HCRYPTKEY hKey = NULL;

			// derive a session key from the hash object
			if (!CryptDeriveKey(hCryptProv, ENCRYPT_ALGORITHM, hHash, KEYLENGTH, &hKey))
			{
				// error during CryptDeriveKey
				sqlite3_result_error(context, "error: CryptDeriveKey()", -1);

				CryptDestroyHash(hHash);
				CryptReleaseContext(hCryptProv, 0);
				return;
			}

			// encrypt
			DWORD length = (DWORD)rawData.length();
			std::string encryptedData(rawData);
			if (!CryptEncrypt(hKey, NULL, TRUE, 0, (unsigned char *)encryptedData.c_str(), &length, encryptedData.length()))
			{
				// error during CryptEncrypt
				sqlite3_result_error(context, "error: CryptEncrypt()", -1);

				CryptDestroyKey(hKey);
				CryptDestroyHash(hHash);
				CryptReleaseContext(hCryptProv, 0);
				return;
			}

			sqlite3_result_blob(context, encryptedData.c_str(), encryptedData.length(), SQLITE_TRANSIENT);

			CryptDestroyKey(hKey);
			CryptDestroyHash(hHash);
			CryptReleaseContext(hCryptProv, 0);
		}
	}
#else
    throw std::runtime_error("not yet implemented");
#endif
}

//------------------------------------------------------------------------

void decrypt(sqlite3_context *context, int argc, sqlite3_value **argv)
{
#if defined(WIN32)
	if (2 != argc)
	{
		sqlite3_result_error(context, "error: Invalid param count", -1);
		return;
	}
	else
	{
		const std::string encryptedData((const char *)sqlite3_value_blob(argv[0]));
		const std::string keyPhrase((const char *)sqlite3_value_text(argv[1]));

		if (!keyPhrase.empty())
		{
			HCRYPTPROV hCryptProv = NULL;

			// get the Handle to the default provider
			if (!CryptAcquireContext(&hCryptProv, NULL, MS_ENHANCED_PROV, PROV_RSA_FULL, 0))
			{
				if (GetLastError() == NTE_BAD_KEYSET)
				{
					if (!CryptAcquireContext(&hCryptProv, NULL, MS_DEF_PROV, PROV_RSA_FULL, CRYPT_NEWKEYSET))
					{
						// CryptAcquireContext() failed.
						sqlite3_result_error(context, "error: CryptAcquireContext()", -1);
						return;
					}
				}
				else
				{
					// CryptAcquireContext() failed.
					sqlite3_result_error(context, "error: CryptAcquireContext())", -1);
					return;
				}
			}

			HCRYPTHASH hHash = NULL;

			// create Hash object
			if (!CryptCreateHash(hCryptProv, CALG_MD5, 0, 0, &hHash) || !hHash)
			{
				// error during CryptCreateHash()
				sqlite3_result_error(context, "error: CryptCreateHash()", -1);

				CryptReleaseContext(hCryptProv, 0);
				return;
			}

			// hash the password
			if (!CryptHashData(hHash, (const unsigned char *)keyPhrase.c_str(), keyPhrase.length(), 0))
			{
				// error during CryptHashData
				sqlite3_result_error(context, "error: CryptHashData()", -1);

				CryptDestroyHash(hHash);
				CryptReleaseContext(hCryptProv, 0);
				return;
			}

			HCRYPTKEY hKey = NULL;

			// derive a session key from the hash object
			if (!CryptDeriveKey(hCryptProv, ENCRYPT_ALGORITHM, hHash, KEYLENGTH, &hKey))
			{
				// error during CryptDeriveKey
				sqlite3_result_error(context, "error: CryptDeriveKey()", -1);

				CryptDestroyHash(hHash);
				CryptReleaseContext(hCryptProv, 0);
				return;
			}

			// decrypt
			DWORD length = (DWORD)encryptedData.length();
			std::string rawData(encryptedData);
			if (!CryptDecrypt(hKey, NULL, TRUE, 0, (unsigned char *)rawData.c_str(), &length))
			{
				// error during CryptEncrypt
				sqlite3_result_error(context, "error: CryptDecrypt()", -1);
				CryptDestroyKey(hKey);
				CryptDestroyHash(hHash);
				CryptReleaseContext(hCryptProv, 0);
				return;
			}

			sqlite3_result_blob(context, rawData.c_str(), rawData.length(), SQLITE_TRANSIENT);

			CryptDestroyKey(hKey);
			CryptDestroyHash(hHash);
			CryptReleaseContext(hCryptProv, 0);
		}
	}
#else
    throw std::runtime_error("not yet implemented");
#endif
}

void create_table(sqlite3 *db)
{
	const std::string sql = "CREATE TABLE user_tbl(id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL UNIQUE, user_name varchar(32) NOT NULL, passwd varch(32) NOT NULL, value smallint DEFAULT 0);";

	char *errMsg = NULL;
	const int rc = sqlite3_exec(db, sql.c_str(), NULL, NULL, &errMsg);
	if (SQLITE_OK != rc)
	{
		std::cerr << "SQL error: " << errMsg << std::endl;
		sqlite3_free(errMsg);
	}
}

void insert_data(sqlite3 *db, const std::string &user_id, const std::string &passwd, const short val, const std::string &encrytionPassPhrase)
{
	std::ostringstream sql;
	//sql << "INSERT INTO user_tbl(id, user_name, passwd, value) Values(" << i+1 << ", '" << user_id.c_str() << "', encrypt('" << passwd.c_str() << "', '" << encrytionPassPhrase.c_str() << "'), " << val << ");";
	sql << "INSERT INTO user_tbl(user_name, passwd, value) Values('" << user_id.c_str() << "', encrypt('" << passwd.c_str() << "', '" << encrytionPassPhrase.c_str() << "'), " << val << ");";

	char *errMsg = NULL;
	const int rc = sqlite3_exec(db, sql.str().c_str(), NULL, NULL, &errMsg);
	if (SQLITE_OK != rc)
	{
		std::cerr << "SQL error: " << errMsg << std::endl;
		sqlite3_free(errMsg);
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_sqlite {

void encryption_decryption()
{
	// open database
	//const std::string databaseName = "database_data\\sqlite\\test.db";  // file db
	const std::string databaseName = ":memory:";  // memory db

	sqlite3 *db = NULL;
	const int rc = sqlite3_open(databaseName.c_str(), &db);
	if (SQLITE_OK != rc)
	{
		std::cerr << "can't open database: " << sqlite3_errmsg(db) << std::endl;
		//sqlite3_close(db);
		return;
	}

	// create functions
	if (SQLITE_OK != sqlite3_create_function(db, "encrypt", -1, SQLITE_UTF8, NULL, &local::encrypt, NULL, NULL) ||
		SQLITE_OK != sqlite3_create_function(db, "decrypt", -1, SQLITE_UTF8, NULL, &local::decrypt, NULL, NULL))
	{
		std::cerr << "can't create function: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_close(db);
		return;
	}

	//
	const std::string encryptionPassPhrase("12345!@#$%");

	//
	local::create_table(db);
	local::insert_data(db, "honggildong", "gildong2", 3, encryptionPassPhrase);
	local::insert_data(db, "sungchunhyang", "chunhyang2", 7, encryptionPassPhrase);

	// encrypt
	{
		std::ostringstream stream;
		stream << "SELECT user_name, passwd FROM user_tbl WHERE passwd = encrypt('" << "gildong2" << "', '" << encryptionPassPhrase << "')";
		//stream << "SELECT user_name, passwd FROM user_tbl";
		const std::string sql(stream.str());

		sqlite3_stmt *statement = NULL;
		if (sqlite3_prepare_v2(db, sql.c_str(), sql.length(), &statement, NULL) == SQLITE_OK)
		{
			const int nRow = sqlite3_data_count(statement);
			const int nCol = sqlite3_column_count(statement);

			const std::string col0name(sqlite3_column_name(statement, 0));
			const std::string col1name(sqlite3_column_name(statement, 1));

			int rowIdx = 0;
			while (SQLITE_ROW == sqlite3_step(statement))
			{
				++rowIdx;

				const std::string col0((char *)sqlite3_column_text(statement, 0));
				const std::string col1((char *)sqlite3_column_text(statement, 1));

				std::cout << "ROW " << rowIdx << " : " << col0name << " => " << col0 << ", " << col1 << std::endl;
			}
		}
		else
			std::cerr << "error: " << sqlite3_errmsg(db) << std::endl;

		sqlite3_finalize(statement);
	}

	// decrypt
	{
		std::ostringstream stream;
		stream << "SELECT user_name, passwd FROM user_tbl WHERE passwd = decrypt('" << "gildong2" << "', '" << encryptionPassPhrase << "')";
		//stream << "SELECT user_name, passwd FROM user_tbl WHERE passwd = '" << "gildong2" << "'";
		const std::string sql(stream.str());

		sqlite3_stmt *statement = NULL;
		if (sqlite3_prepare_v2(db, sql.c_str(), sql.length(), &statement, NULL) == SQLITE_OK)
		{
			const int nRow = sqlite3_data_count(statement);
			const int nCol = sqlite3_column_count(statement);

			const std::string col0name(sqlite3_column_name(statement, 0));
			const std::string col1name(sqlite3_column_name(statement, 1));

			int rowIdx = 0;
			while (SQLITE_ROW == sqlite3_step(statement))
			{
				++rowIdx;

				const std::string col0((char *)sqlite3_column_text(statement, 0));
				const std::string col1((char *)sqlite3_column_text(statement, 1));

				std::cout << "ROW " << rowIdx << " : " << col0name << " => " << col0 << ", " << col1 << std::endl;
			}
		}
		else
			std::cerr << "error: " << sqlite3_errmsg(db) << std::endl;

		sqlite3_finalize(statement);
	}

	// close database
	sqlite3_close(db);
}

}  // namespace my_sqlite
