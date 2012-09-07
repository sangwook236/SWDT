using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace sqlite
{
    using System.Data;
    using System.Data.SQLite;
    using System.Collections;
    using System.ComponentModel;

    class UsingAdoNet
    {
        public static void runTests()
        {
            // populate a DataSet class with a DataAdapter.
            DataSet colorDataSet = populateColorDataSet();
            if (null == colorDataSet)
            {
                Console.WriteLine("data set creation error");
                return;
            }

            Console.WriteLine(">>>>> after selecting ...");
            DataSet selectedDataSet = runSelectOperation(colorDataSet);
            printDataSet(selectedDataSet);

            Console.WriteLine("\n>>>>> after inserting ...");
            runInsertOperation(colorDataSet);
            printDataSet(colorDataSet);

            Console.WriteLine("\n>>>>> after updating ...");
            runUpdateOperation(colorDataSet);
            printDataSet(colorDataSet);

            Console.WriteLine("\n>>>>> after deleting ...");
            runDeleteOperation(colorDataSet);
            printDataSet(colorDataSet);
        }

        static void printDataSet(DataSet colorDataSet)
        {
#if false
            DataView dataView = colorDataSet.Tables[colorsTable_].DefaultView;
            foreach (DataRowView row in dataView)
            {
                long? id = row[0] as long?;
                string name = row[1] as string;
                string hexchars = row[2] as string;
                string description = row[3] as string;
                long? hexcode = row[4] as long?;
                Console.WriteLine("ID: {0}, Name: {1}, HexChars: {2}, Description: {3}, HexCode: {4}",
                    (null == id ? "null" : id.ToString()),
                    (null == name ? "null" : name),
                    (null == hexchars ? "null" : hexchars),
                    (null == description ? "null" : description),
                    (null == hexcode ? "null" : string.Format("0x{0:X}", hexcode))
                );
            }
#else
            foreach (DataRow row in colorDataSet.Tables[colorsTable_].Rows)
            {
                long? id = row[0] as long?;
                string name = row[1] as string;
                string hexchars = row[2] as string;
                string description = row[3] as string;
                long? hexcode = row[4] as long?;
                Console.WriteLine("ID: {0}, Name: {1}, HexChars: {2}, Description: {3}, HexCode: {4}",
                    (null == id ? "null" : id.ToString()),
                    (null == name ? "null" : name),
                    (null == hexchars ? "null" : hexchars),
                    (null == description ? "null" : description),
                    (null == hexcode ? "null" : string.Format("0x{0:X}", hexcode))
                );
            }
#endif
        }

        static DataSet populateColorDataSet()
        {
            try
            {
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    connection.Open();

                    string sql = string.Format("SELECT * FROM {0}", colorsTable_);
                    SQLiteDataAdapter adapter = new SQLiteDataAdapter(sql, connection);
                    DataSet ds = new DataSet();

                    adapter.Fill(ds, colorsTable_);
                    //adapter.Fill(ds);

                    return ds;
                }
            }
            catch (System.Data.SQLite.SQLiteException ex)
            {
                Console.WriteLine("SQLite error: {0}", ex.Message);
            }

            return null;
        }

        static DataSet runSelectOperation(DataSet colorDataSet)
        {
            try
            {
                DataTable table = colorDataSet.Tables[colorsTable_];

                IEnumerable<DataRow> queriedRows =
                    from c in table.AsEnumerable()
                    where c.Field<long>("id") >= 2
                    select c;

                //int count = queriedRows.Count<DataRow>();

                //
                DataTable boundTable = queriedRows.CopyToDataTable<DataRow>();
                boundTable.TableName = colorsTable_;

                //DataSet ds = boundTable.DataSet;  // null
                //DataSet ds = colorDataSet.Clone();  // copy schemas
                DataSet ds = new DataSet();
                ds.Tables.Add(boundTable);

                //int num = ds.Tables.Count;

                return ds;
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine("DB query error: {0}", ex.Message);
            }

            return null;
        }

        static void runInsertOperation(DataSet colorDataSet)
        {
            try
            {
                // insert a new row to a DataTable.
                {
                    DataTable table = colorDataSet.Tables[colorsTable_];

                    long index = (from c in table.AsEnumerable()
                                  select c.Field<long>("id")).Max() + 1;

#if false
                    //colorDataSet.Tables[colorsTable_].Rows.Add(new object[] { null, "magenta", "ff00ff", "it's a magenta", 0xFF00FF });  // not correctly working
                    colorDataSet.Tables[colorsTable_].Rows.Add(new object[] { index, "magenta", "ff00ff", "it's a magenta", 0xFF00FF });
#else
                    DataRow row = colorDataSet.Tables[colorsTable_].NewRow();
                    //row["id"] = System.DBNull.Value;  // not correctly working
                    row["id"] = index;
                    row["name"] = "magenta";
                    row["hexchars"] = "ff00ff";
                    row["description"] = "it's a magenta";
                    row["hexcode"] = 0xFF00FF;

                    colorDataSet.Tables[colorsTable_].Rows.Add(row);
#endif
                }

                // update with a DataAdapter: inserting.
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    connection.Open();

                    SQLiteDataAdapter adapter = new SQLiteDataAdapter();
                    adapter.InsertCommand = generateInsertCommand(connection);

                    adapter.Update(colorDataSet, colorsTable_);
                    //adapter.Update(colorDataSet.GetChanges());  // run-time error.
                    //adapter.Update(colorDataSet.GetChanges(), colorsTable_);  // an exception occurred if there is no change.
                }

                //colorDataSet.Merge();
                colorDataSet.AcceptChanges();
            }
            catch (System.Data.SQLite.SQLiteException ex)
            {
                Console.WriteLine("SQLite error: {0}", ex.Message);
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine("DB query error: {0}", ex.Message);
            }
            catch (System.Data.DBConcurrencyException ex)
            {
                Console.WriteLine("DB concurrency error: {0}", ex.Message);
                throw;
            }

            //
            try
            {
                // insert a new row to a DataTable.
                {
                    DataTable table = colorDataSet.Tables[colorsTable_];

                    long index = (from d in table.AsEnumerable()
                                  select d.Field<long>("id")).Max() + 1;

                    for (int i = 0; i < 16; ++i)
                    {
#if false
                        //colorDataSet.Tables[colorsTable_].Rows.Add(new object[] { null, string.Format("Color #{0:D2}", i + 1), string.Format("{0:X6}", i + 1), null, null });  // not correctly working
                        colorDataSet.Tables[colorsTable_].Rows.Add(new object[] { index + i, string.Format("Color #{0:D2}", i + 1), string.Format("{0:X6}", i + 1), null, null });
#else
                        DataRow row = colorDataSet.Tables[colorsTable_].NewRow();
                        //row["id"] = System.DBNull.Value;  // not correctly working
                        row["id"] = index + i;
                        row["name"] = string.Format("Color #{0:D2}", i + 1);
                        row["hexchars"] = string.Format("{0:X6}", i + 1);
                        row["description"] = System.DBNull.Value;
                        row["hexcode"] = System.DBNull.Value;

                        colorDataSet.Tables[colorsTable_].Rows.Add(row);
#endif
                    }
                }

                // update with a DataAdapter: inserting.
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    connection.Open();

                    SQLiteDataAdapter adapter = new SQLiteDataAdapter();
                    adapter.InsertCommand = generateInsertCommand(connection);

                    adapter.Update(colorDataSet, colorsTable_);
                    //adapter.Update(colorDataSet.GetChanges());  // run-time error.
                    //adapter.Update(colorDataSet.GetChanges(), colorsTable_);  // an exception occurred if there is no change.
                }

                colorDataSet.AcceptChanges();
            }
            catch (System.Data.SQLite.SQLiteException ex)
            {
                Console.WriteLine("SQLite error: {0}", ex.Message);
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine("DB query error: {0}", ex.Message);
            }
            catch (System.Data.DBConcurrencyException ex)
            {
                Console.WriteLine("DB concurrency error: {0}", ex.Message);
                throw;
            }
        }

        static void runUpdateOperation(DataSet colorDataSet)
        {
            try
            {
                // update an existing row within a DataTable.
                {
                    DataTable table = colorDataSet.Tables[colorsTable_];

                    DataRow queriedRow = (from c in table.AsEnumerable()
                                          where c.Field<string>("name") == "Color #01"
                                          select c).First();

                    queriedRow["hexchars"] = "0F0F0F";
                }

                // update with a DataAdapter: updating.
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    connection.Open();

                    SQLiteDataAdapter adapter = new SQLiteDataAdapter();
                    adapter.UpdateCommand = generateUpdateCommand(connection);

                    adapter.Update(colorDataSet, colorsTable_);
                    //adapter.Update(colorDataSet.GetChanges());  // run-time error.
                    //adapter.Update(colorDataSet.GetChanges(), colorsTable_);  // an exception occurred if there is no change.
                }

                colorDataSet.AcceptChanges();
            }
            catch (System.Data.SQLite.SQLiteException ex)
            {
                Console.WriteLine("SQLite error: {0}", ex.Message);
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine("DB query error: {0}", ex.Message);
            }
            catch (System.Data.DBConcurrencyException ex)
            {
                Console.WriteLine("DB concurrency error: {0}", ex.Message);
                throw;
            }

            try
            {
                // update an existing row within a DataTable.
                {
                    DataTable table = colorDataSet.Tables[colorsTable_];

                    DataRow queriedRow = (from c in table.AsEnumerable()
                                          where c.Field<string>("name") == "Color #20"
                                          select c).First();

                    queriedRow["hexchars"] = "0F0F0F";

                    // update with a DataAdapter: updating.
                    using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                    {
                        connection.Open();

                        SQLiteDataAdapter adapter = new SQLiteDataAdapter();
                        adapter.UpdateCommand = generateUpdateCommand(connection);

                        adapter.Update(colorDataSet, colorsTable_);
                        //adapter.Update(colorDataSet.GetChanges());  // run-time error.
                        //adapter.Update(colorDataSet.GetChanges(), colorsTable_);  // an exception occurred if there is no change.
                    }

                    colorDataSet.AcceptChanges();
                }
            }
            catch (System.Data.SQLite.SQLiteException ex)
            {
                Console.WriteLine("SQLite error: {0}", ex.Message);
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine("DB query error: {0}", ex.Message);
            }
            catch (System.Data.DBConcurrencyException ex)
            {
                Console.WriteLine("DB concurrency error: {0}", ex.Message);
                throw;
            }
        }

        static void runDeleteOperation(DataSet colorDataSet)
        {
            try
            {
                // delete a row within a DataTable.
                {
                    DataTable table = colorDataSet.Tables[colorsTable_];

                    IEnumerable<DataRow> queriedRows = from c in table.AsEnumerable()
                                                       where c.Field<long>("id") > 3
                                                       select c;

                    foreach (DataRow row in queriedRows)
                        row.Delete();
                }

                // update with a DataAdapter: deleting.
                using (SQLiteConnection connection = new SQLiteConnection(connectionStr_))
                {
                    connection.Open();

                    SQLiteDataAdapter adapter = new SQLiteDataAdapter();
                    adapter.DeleteCommand = generateDeleteCommand(connection);

                    adapter.Update(colorDataSet, colorsTable_);
                    //adapter.Update(colorDataSet.GetChanges());  // run-time error.
                    //adapter.Update(colorDataSet.GetChanges(), colorsTable_);  // an exception occurred if there is no change.
                }

                colorDataSet.AcceptChanges();
            }
            catch (System.Data.SQLite.SQLiteException ex)
            {
                Console.WriteLine("SQLite error: {0}", ex.Message);
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine("DB query error: {0}", ex.Message);
            }
            catch (System.Data.DBConcurrencyException ex)
            {
                Console.WriteLine("DB concurrency error: {0}", ex.Message);
                throw;
            }
        }

        static SQLiteCommand generateInsertCommand(SQLiteConnection connection)
        {
            SQLiteCommand command = new SQLiteCommand(connection);

            command.CommandText = string.Format("INSERT INTO {0}(id, name, hexchars, description, hexcode) VALUES(?, ?, ?, ?, ?)", colorsTable_);
            //command.CommandText = string.Format("INSERT INTO {0}(name, hexchars, description, hexcode) VALUES(?, ?, ?, ?)", colorsTable_);
            command.Parameters.Add(new SQLiteParameter(DbType.Int64, "id"));
            command.Parameters.Add(new SQLiteParameter(DbType.StringFixedLength, 128, "name"));
            command.Parameters.Add(new SQLiteParameter(DbType.StringFixedLength, 6, "hexchars"));
            command.Parameters.Add(new SQLiteParameter(DbType.StringFixedLength, 128, "description"));
            command.Parameters.Add(new SQLiteParameter(DbType.Int64, "hexcode"));
            command.UpdatedRowSource = UpdateRowSource.OutputParameters;
            //command.UpdatedRowSource = UpdateRowSource.None;

            return command;
        }

        static SQLiteCommand generateUpdateCommand(SQLiteConnection connection)
        {
            SQLiteCommand command = new SQLiteCommand(connection);

#if false
            command.CommandText = string.Format("UPDATE {0} SET hexchars = ? WHERE name = ?", colorsTable_);
            command.Parameters.Add(new SQLiteParameter(DbType.StringFixedLength, 6, "hexchars"));
            command.Parameters.Add(new SQLiteParameter(DbType.StringFixedLength, 128, "name"));
#else
            command.CommandText = string.Format("UPDATE {0} SET name = ?, hexchars = ?, description = ?, hexcode = ? WHERE id = ?", colorsTable_);
            command.Parameters.Add(new SQLiteParameter(DbType.StringFixedLength, 128, "name"));
            command.Parameters.Add(new SQLiteParameter(DbType.StringFixedLength, 6, "hexchars"));
            command.Parameters.Add(new SQLiteParameter(DbType.StringFixedLength, 128, "description"));
            command.Parameters.Add(new SQLiteParameter(DbType.Int64, "hexcode"));
            command.Parameters.Add(new SQLiteParameter(DbType.Int64, "id"));
#endif
            //command.UpdatedRowSource = UpdateRowSource.OutputParameters;
            command.UpdatedRowSource = UpdateRowSource.None;

            return command;
        }

        static SQLiteCommand generateDeleteCommand(SQLiteConnection connection)
        {
            SQLiteCommand command = new SQLiteCommand(connection);

            //command.CommandText = string.Format("DELETE FROM {0} WHERE id > ?", colorsTable_);  // run-time error: DB concurrency error.
            command.CommandText = string.Format("DELETE FROM {0} WHERE id == ?", colorsTable_);
            //command.CommandText = string.Format("DELETE FROM {0}", colorsTable_);  // delete all the record in a ColorsTable
            command.Parameters.Add(new SQLiteParameter(DbType.Int64, "id"));
            //command.UpdatedRowSource = UpdateRowSource.OutputParameters;
            command.UpdatedRowSource = UpdateRowSource.None;

            return command;
        }

        //private static string connectionStr_ = @"Data Source=""..\data\sqlite_data\sqlite3_test.db"";Version=3;";
        private static string connectionStr_ = Properties.Settings.Default.TestDatabase;
        private static string colorsTable_ = "Colors";
    }
}
