package sqlite;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class SQLite_Main {

	public static void run(String[] args) throws ClassNotFoundException
	{
	    // Load the sqlite-JDBC driver using the current class loader.
	    Class.forName("org.sqlite.JDBC");

	    Connection connection = null;
	    try
	    {
	      // Create a database connection.
	      connection = DriverManager.getConnection("jdbc:sqlite:db/sqlite_test.s3db");
	      Statement statement = connection.createStatement();
	      statement.setQueryTimeout(30);  // Set timeout to 30 sec.

	      statement.executeUpdate("drop table if exists person");
	      statement.executeUpdate("create table person (id integer, name string)");
	      statement.executeUpdate("insert into person values(1, 'leo')");
	      statement.executeUpdate("insert into person values(2, 'yui')");
	      ResultSet rs = statement.executeQuery("select * from person");

	      while (rs.next())
	      {
	    	  // Read the result set.
	    	  System.out.println("name = " + rs.getString("name"));
	    	  System.out.println("id = " + rs.getInt("id"));
	      }
	    }
	    catch (SQLException ex)
	    {
	    	// If the error message is "out of memory", 
	    	// It probably means no database file is found.
	    	System.err.println(ex.getMessage());
	    }
	    finally
	    {
	    	try
	    	{
	    		if (connection != null)
	    			connection.close();
	    	}
	    	catch (SQLException ex)
	    	{
	    		// Connection close failed.
	    		System.err.println(ex);
	    	}
	    }
	}

}
