//#include "stdafx.h"
#if defined(__linux) || defined(__linux__) || defined(linux) || defined(__unix) || defined(__unix__) || defined(unix)
#include <sqlite3.h>
#else
#include <sqlite/sqlite3.h>
#endif
#include <sstream>
#include <iostream>


namespace {
namespace local {

int callback(void *param, int argc, char **argv, char **colName)
{
	for (int i = 0; i < argc; ++i)
		std::cout << colName[i] << " = " << (argv[i] ? argv[i] : "NULL") << std::endl;
	std::cout << std::endl;
	return 0;
}

void create_table(sqlite3 *db)
{
	const std::string sql = "CREATE TABLE tbl_Account(Id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL UNIQUE, Account varchar(32), Grade smallint DEFAULT 1);";
	char *errMsg = NULL;
	const int rc = sqlite3_exec(db, sql.c_str(), NULL, NULL, &errMsg);
	if (SQLITE_OK != rc)
	{
		std::cerr << "SQL error: " << errMsg << std::endl;
		sqlite3_free(errMsg);
	}
}

void insert_data_1(sqlite3 *db)
{
	for (int i = 0; i < 10; ++i)
	{
		std::ostringstream sql;
		//sql << "INSERT INTO tbl_Account(Id, Account, Grade) Values(" << i+1 << ", 'dummy" << i << "', 1);";  // it's also working
		sql << "INSERT INTO tbl_Account(Account, Grade) Values(" << "'dummy" << i << "', 1);";
		char *errMsg = NULL;
		const int rc = sqlite3_exec(db, sql.str().c_str(), NULL, NULL, &errMsg);
		if (SQLITE_OK != rc)
		{
			std::cerr << "SQL error: " << errMsg << std::endl;
			sqlite3_free(errMsg);
		}
	}
}

void insert_data_2(sqlite3 *db)
{
	sqlite3_stmt *stmts = NULL;

	const std::string sql = "insert into tbl_Account(id, account, grade) values(?, ?, ?)";
	if (SQLITE_OK == sqlite3_prepare_v2(db, sql.c_str(), -1, &stmts, NULL))  // UTF-8 encoded
	//if (SQLITE_OK == sqlite3_prepare16_v2(db, sql.c_str(), -1, &stmts, NULL))  // UTF-16 encoded
	{
		if (SQLITE_OK == sqlite3_bind_int(stmts, 1, 12) &&
			SQLITE_OK == sqlite3_bind_text(stmts, 2, "dummy12", -1, SQLITE_TRANSIENT) &&
			SQLITE_OK == sqlite3_bind_int(stmts, 3, 2))
		{
			while (SQLITE_ROW == sqlite3_step(stmts))
			{
			}

			const int lastRow = (int)sqlite3_last_insert_rowid(db);
			std::cout << "last inserted row ID: " << lastRow << std::endl;

			sqlite3_reset(stmts);
		}
		else
		{
			const char *errMsg = sqlite3_errmsg(db);
			std::cerr << "SQL error: " << errMsg << std::endl;
		}
	}
	else
	{
		const char *errMsg = sqlite3_errmsg(db);
		std::cerr << "SQL error: " << errMsg << std::endl;
	}

	sqlite3_finalize(stmts);
}

void delete_data_1(sqlite3 *db)
{
	std::ostringstream sql;
	sql << "DELETE FROM tbl_Account WHERE id = 8";
	char *errMsg = NULL;
	const int rc = sqlite3_exec(db, sql.str().c_str(), NULL, NULL, &errMsg);
	if (SQLITE_OK != rc)
	{
		std::cerr << "SQL error: " << errMsg << std::endl;
		sqlite3_free(errMsg);
	}
}

void delete_data_2(sqlite3 *db)
{
	sqlite3_stmt *stmts = NULL;

	const std::string sql = "delete from tbl_Account where id = ?";
	if (SQLITE_OK == sqlite3_prepare_v2(db, sql.c_str(), -1, &stmts, NULL))  // UTF-8 encoded
	//if (SQLITE_OK == sqlite3_prepare16_v2(db, sql.c_str(), -1, &stmts, NULL))  // UTF-16 encoded
	{
		if (SQLITE_OK == sqlite3_bind_int(stmts, 1, 4))
		{
			while (SQLITE_ROW == sqlite3_step(stmts))
			{
			}

			sqlite3_reset(stmts);
		}
		else
		{
			const char *errMsg = sqlite3_errmsg(db);
			std::cerr << "SQL error: " << errMsg << std::endl;
		}
	}
	else
	{
		const char *errMsg = sqlite3_errmsg(db);
		std::cerr << "SQL error: " << errMsg << std::endl;
	}

	sqlite3_finalize(stmts);
}

void select_data_1(sqlite3 *db)
{
	const std::string sql = "SELECT * FROM tbl_Account";
	char *errMsg = NULL;
	const int rc = sqlite3_exec(db, sql.c_str(), callback, NULL, &errMsg);
	if (SQLITE_OK != rc)
	{
		std::cerr << "SQL error: " << errMsg << std::endl;
		sqlite3_free(errMsg);
	}
}

void select_data_2(sqlite3 *db)
{
	sqlite3_stmt *selectedStmts = NULL;

	const std::string sql = "SELECT Id, Account FROM tbl_Account";
	if (SQLITE_OK == sqlite3_prepare_v2(db, sql.c_str(), -1, &selectedStmts, NULL))  // UTF-8 encoded
	//if (SQLITE_OK == sqlite3_prepare16_v2(db, sql.c_str(), -1, &selectedStmts, NULL))  // UTF-16 encoded
	{
		std::cout << "the number of data: " << sqlite3_data_count(selectedStmts) << std::endl;  // Oops !!!

		while (SQLITE_ROW == sqlite3_step(selectedStmts))
		{
			const int id = sqlite3_column_int(selectedStmts, 0);
			const unsigned char *account = sqlite3_column_text(selectedStmts, 1);

			std::cout << "id: " << id << ", account: " << account << std::endl;

			//sqlite3_free((void *)account);
		}

		sqlite3_reset(selectedStmts);
	}
	else
	{
		const char *errMsg = sqlite3_errmsg(db);
		std::cerr << "SQL error: " << errMsg << std::endl;
		//sqlite3_free((void *)errMsg);
	}

	sqlite3_finalize(selectedStmts);
}

}  // namespace local
}  // unnamed namespace

namespace my_sqlite {

void basic()
{
	// open database
	//const std::string databaseName = "data/database/sqlite/test.db";  // file db
	const std::string databaseName = ":memory:";  // memory db

	sqlite3 *db = NULL;
	const int rc = sqlite3_open(databaseName.c_str(), &db);
	if (SQLITE_OK != rc)
	{
		std::cerr << "can't open database: " << sqlite3_errmsg(db) << std::endl;
		//sqlite3_close(db);
		return;
	}

	// create table
	local::create_table(db);

	// insert data
	local::insert_data_1(db);
	local::insert_data_2(db);

	// select data
	local::select_data_1(db);
	local::select_data_2(db);

	// delete data
	local::delete_data_1(db);
	local::delete_data_2(db);

	local::select_data_2(db);

	// close database
	sqlite3_close(db);
}

}  // namespace my_sqlite
