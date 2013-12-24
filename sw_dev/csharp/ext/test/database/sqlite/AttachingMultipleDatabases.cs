using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace database.sqlite
{
    using System.Data.SQLite;
    using System.Data;

    class AttachingMultipleDatabases
    {
        public static void runTests()
        {
            // [ref] http://www.sqlite.org/c3ref/open.html
            // [ref] http://sqlite.phxsoftware.com/forums/t/130.aspx

            string[] db_files = {
                ".\\data\\database\\sqlite3_test1.db",
                ".\\data\\database\\sqlite3_test2.db",
            };

            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    connection.Open();

                    for (int i = 0; i < db_files.Length; ++i)
                    {
                        using (SQLiteCommand cmd = new SQLiteCommand(string.Format("ATTACH DATABASE '{0}' AS attched_db_{1}", db_files[i], i), connection))
                            cmd.ExecuteNonQuery();
                    }

#if false
                    {
                        string sql = string.Format("SELECT * FROM {0}", colorsTable_);
                        using (SQLiteCommand cmd = new SQLiteCommand(sql, connection))
                        {
                            SQLiteDataReader reader = null;
                            try
                            {
                                //reader = cmd.ExecuteReader(CommandBehavior.CloseConnection);
                                reader = cmd.ExecuteReader(CommandBehavior.SequentialAccess);
                                while (reader.Read())
                                    Console.WriteLine("ID: {0}, Name: {1}, HexChars: {2}, Description: {3}, HexCode: {4:X}", reader["id"], reader["name"], reader["hexchars"], (reader.IsDBNull(3) ? "null" : reader["description"]), (reader.IsDBNull(4) ? "null" : reader["hexcode"].ToString()));
                            }
                            finally
                            {
                                reader.Close();
                            }
                        }
                    }

                    {
                        string sql = string.Format("SELECT * FROM attched_db_{0}.{1}", 0, colorsTable_);
                        using (SQLiteCommand cmd = new SQLiteCommand(sql, connection))
                        {
                            SQLiteDataReader reader = null;
                            try
                            {
                                reader = cmd.ExecuteReader(CommandBehavior.SequentialAccess);
                                while (reader.Read())
                                    Console.WriteLine("ID: {0}, Name: {1}, HexChars: {2}, Description: {3}, HexCode: {4}", reader["id"], reader["name"], reader["hexchars"], (reader.IsDBNull(3) ? "null" : reader["description"]), (reader.IsDBNull(4) ? "null" : string.Format("0x{0:X}", (long)reader["hexcode"])));
                            }
                            finally
                            {
                                reader.Close();
                            }
                        }
                    }
#else

                    {
#if true
                        StringBuilder builder = new StringBuilder();
                        builder.AppendFormat("SELECT * FROM {0}", colorsTable_);
                        for (int i = 0; i < db_files.Length; ++i)
                        {
                            builder.AppendFormat(" UNION SELECT * FROM attched_db_{1}.{0}", colorsTable_, i);
                        }

                        string sql = builder.ToString();
#else
                        string sql = string.Format("SELECT * FROM {0} UNION SELECT * FROM attched_db_{1}.{0}", colorsTable_, 0);
#endif
                        using (SQLiteCommand cmd = new SQLiteCommand(sql, connection))
                        {
                            SQLiteDataReader reader = null;
                            try
                            {
                                //reader = cmd.ExecuteReader(CommandBehavior.CloseConnection);
                                reader = cmd.ExecuteReader(CommandBehavior.SequentialAccess);
                                while (reader.Read())
                                    Console.WriteLine("ID: {0}, Name: {1}, HexChars: {2}, Description: {3}, HexCode: {4}", reader["id"], reader["name"], reader["hexchars"], (reader.IsDBNull(3) ? "null" : reader["description"]), (reader.IsDBNull(4) ? "null" : string.Format("0x{0:X}", (long)reader["hexcode"])));
                            }
                            finally
                            {
                                reader.Close();
                            }
                        }
                    }
#endif

                    {
                        // [ref] http://longweekendmobile.com/2010/05/29/how-to-attach-multiple-sqlite-databases-together/

                        string sql = string.Format("SELECT name FROM attched_db_{0}.sqlite_master WHERE type='{1}'", 0, colorsTable_);
                        using (SQLiteCommand cmd = new SQLiteCommand(sql, connection))
                        {
                            SQLiteDataReader reader = null;
                            try
                            {
                                reader = cmd.ExecuteReader(CommandBehavior.SequentialAccess);
                                while (reader.Read())
                                {
                                    Console.WriteLine(reader["name"]);
                                }
                            }
                            finally
                            {
                                reader.Close();
                            }
                        }
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }
        }

        //private static string connectionStr_ = @"Data Source=""..\data\database\sqlite\sqlite3_test.db"";Version=3;";
        private static string connectionStr_ = Properties.Settings.Default.TestDatabase;
        //private static string connection2Str_ = @"Data Source=""..\data\database\sqlite\sqlite3_test2.db"";Version=3;";
        private static string connection2Str_ = Properties.Settings.Default.Test2Database;
        private static string colorsTable_ = "Colors";
    }
}
