using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace sqlite
{
    using System.Data;
    using System.Data.SQLite;

    class BasicOperation
    {
        public static void runTests()
        {
            Console.WriteLine(">>>>> after inserting ...");
            runInsertOperation();
            runSelectOperation();

            Console.WriteLine("\n>>>>> after updating ...");
            runUpdateOperation();
            runSelectOperation();

            Console.WriteLine("\n>>>>> after deleting ...");
            runDeleteOperation();
            runSelectOperation();
        }

        static void runSelectOperation()
        {
            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    connection.Open();

                    string sql = string.Format("SELECT * FROM {0}", colorsTable_);
                    using (SQLiteCommand cmd = new SQLiteCommand(sql, connection))
                    {
                        SQLiteDataReader reader = null;
                        try
                        {
                            reader = cmd.ExecuteReader(CommandBehavior.CloseConnection);
                            while (reader.Read())
                            {
#if true
                                long id = reader.GetInt64(0);
                                string name = reader.GetString(1);
                                string hexchars = reader.GetString(2);
                                string description = reader.IsDBNull(3) ? null : reader.GetString(3);
                                long? hexcode = reader.IsDBNull(4) ? (long?)null : reader.GetInt64(4);
                                Console.WriteLine("ID: {0}, Name: {1}, HexChars: {2}, Description: {3}, HexCode: {4}", id, name, hexchars, (null == description ? "null" : description), (null == hexcode ? "null" : string.Format("0x{0:X}", hexcode)));
#else
                                Console.WriteLine("ID: {0}, Name: {1}, HexChars: {2}, Description: {3}, HexCode: {4}", reader["id"], reader["name"], reader["hexchars"], (reader.IsDBNull(3) ? "null" : reader["description"]), (reader.IsDBNull(4) ? "null" : string.Format("0x{0:X}", (long)reader["hexcode"])));
#endif
                            }
                        }
                        finally
                        {
                            reader.Close();
                        }
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }
        }

        static void runInsertOperation()
        {
            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    connection.Open();

                    using (SQLiteTransaction transaction = connection.BeginTransaction())
                    {
                        try
                        {
                            using (SQLiteCommand command = new SQLiteCommand(connection))
                            {
                                //string sql = string.Format("INSERT INTO {0}(id, name, hexchars, description, hexcode) VALUES({1}, '{2}', '{3}', '{4}', {5})", colorsTable_, 4, "magenta", "ff00ff", "it is a magenta", 0xFF00FF);  // it's also working
                                string sql = string.Format("INSERT INTO {0}(name, hexchars, description, hexcode) VALUES('{1}', '{2}', '{3}', {4})", colorsTable_, "magenta", "ff00ff", "it is a magenta", 0xFF00FF);

                                command.CommandText = sql;
                                command.ExecuteNonQuery();
                            }

                            transaction.Commit();
                        }
                        catch (System.Data.SQLite.SQLiteException e)
                        {
                            Console.WriteLine("SQLite error: {0}", e.Message);
                            transaction.Rollback();
                            //throw;
                        }
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }

            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    connection.Open();

                    using (SQLiteTransaction transaction = connection.BeginTransaction())
                    {
                        try
                        {
                            using (SQLiteCommand command = new SQLiteCommand(connection))
                            {
                                SQLiteParameter param1 = new SQLiteParameter(DbType.StringFixedLength, 128);
                                SQLiteParameter param2 = new SQLiteParameter(DbType.StringFixedLength, 6);
                                SQLiteParameter param3 = new SQLiteParameter(DbType.StringFixedLength, 128);
                                SQLiteParameter param4 = new SQLiteParameter(DbType.Int64);

                                string sql = string.Format("INSERT INTO {0}(name, hexchars, description, hexcode) VALUES(?, ?, ?, ?)", colorsTable_);

                                command.CommandText = sql;
                                command.Parameters.Add(param1);
                                command.Parameters.Add(param2);
                                command.Parameters.Add(param3);
                                command.Parameters.Add(param4);

                                for (int i = 0; i < 16; ++i)
                                {
                                    param1.Value = string.Format("Color #{0:D2}", i + 1);
                                    param2.Value = string.Format("{0:X6}", i + 1);
                                    param3.Value = null;
                                    param4.Value = null;
                                    command.ExecuteNonQuery();
                                }
                            }

                            transaction.Commit();
                        }
                        catch (System.Data.SQLite.SQLiteException e)
                        {
                            Console.WriteLine("SQLite error: {0}", e.Message);
                            transaction.Rollback();
                            //throw;
                        }
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }
        }

        static void runUpdateOperation()
        {
            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    connection.Open();

                    using (SQLiteTransaction transaction = connection.BeginTransaction())
                    {
                        try
                        {
                            using (SQLiteCommand command = new SQLiteCommand(connection))
                            {
                                string sql = string.Format("UPDATE {0} SET hexchars = '0F0F0F' WHERE name = 'Color #01'", colorsTable_);

                                command.CommandText = sql;
                                command.ExecuteNonQuery();
                            }

                            transaction.Commit();
                        }
                        catch (System.Data.SQLite.SQLiteException e)
                        {
                            Console.WriteLine("SQLite error: {0}", e.Message);
                            transaction.Rollback();
                            //throw;
                        }
                    }

                    using (SQLiteTransaction transaction = connection.BeginTransaction())
                    {
                        try
                        {
                            using (SQLiteCommand command = new SQLiteCommand(connection))
                            {
                                string sql = string.Format("UPDATE {0} SET hexchars = '0F0F0F' WHERE name = 'Color #20'", colorsTable_);  // a new record is not inserted.

                                command.CommandText = sql;
                                command.ExecuteNonQuery();
                            }

                            transaction.Commit();
                        }
                        catch (System.Data.SQLite.SQLiteException e)
                        {
                            Console.WriteLine("SQLite error: {0}", e.Message);
                            transaction.Rollback();
                            //throw;
                        }
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }
        }

        static void runDeleteOperation()
        {
            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    connection.Open();

                    using (SQLiteTransaction transaction = connection.BeginTransaction())
                    {
                        try
                        {
                            using (SQLiteCommand command = new SQLiteCommand(connection))
                            {
                                string sql = string.Format("DELETE FROM {0} WHERE id > 3", colorsTable_);
                                //string sql = string.Format("DELETE FROM {0}", colorsTable_);  // delete all the record in a ColorsTable

                                command.CommandText = sql;
                                command.ExecuteNonQuery();
                            }

                            transaction.Commit();
                        }
                        catch (System.Data.SQLite.SQLiteException e)
                        {
                            Console.WriteLine("SQLite error: {0}", e.Message);
                            transaction.Rollback();
                            //throw;
                        }
                    }
                }
            }
            catch (System.Data.SQLite.SQLiteException e)
            {
                Console.WriteLine("SQLite error: {0}", e.Message);
            }
        }

        //private static string connectionStr_ = @"Data Source=""..\data\sqlite_data\sqlite3_test.db"";Version=3;";
        private static string connectionStr_ = Properties.Settings.Default.TestDatabase;
        private static string colorsTable_ = "Colors";
    }
}
