package solr;

import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.beans.Field;
import org.apache.solr.client.solrj.impl.HttpSolrClient;
import org.apache.solr.client.solrj.impl.XMLResponseParser;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrDocumentList;
import org.apache.solr.common.SolrInputDocument;

public class Solr_Main {

	// REF [site] >> https://lucene.apache.org/solr/guide/7_2/using-solrj.html
	// REF [site] >> https://lucene.apache.org/solr/guide/6_6/using-solrj.html
	// REF [site] >> http://lucene.apache.org/solr/guide/7_2/solr-tutorial.html
	public static void run(String[] args)
	{
		// REF [site] >> http://www.baeldung.com/apache-solrj
		indexDocuments();
		indexWithBeans();
		queryIndexedDocumentsByFieldAndId();
		deleteDocuments();
	}

	private static void indexDocuments()
	{
		try
		{
			final String urlString = "http://localhost:8983/solr/techproducts";
			final HttpSolrClient solrClient = new HttpSolrClient.Builder(urlString).build();
			solrClient.setParser(new XMLResponseParser());
	
			final SolrInputDocument document = new SolrInputDocument();
			document.addField("id", "123456");
			document.addField("name", "Kenmore Dishwasher");
			document.addField("price", "599.99");
			solrClient.add(document);
			solrClient.commit();
		}
		catch (SolrServerException ex)
		{
			System.err.println("SolrServerException occurred: " + ex.toString());
		}
		catch (java.io.IOException ex)
		{
			System.err.println("IOException occurred: " + ex.toString());
		}
	}
	
	private static void indexWithBeans()
	{
		try
		{
			final String urlString = "http://localhost:8983/solr/techproducts";
			final HttpSolrClient solrClient = new HttpSolrClient.Builder(urlString).build();
			solrClient.setParser(new XMLResponseParser());
	
			solrClient.addBean(new ProductBean("888", "Apple iPhone 6s", "299.99"));
			solrClient.commit();
		}
		catch (SolrServerException ex)
		{
			System.err.println("SolrServerException occurred: " + ex.toString());
		}
		catch (java.io.IOException ex)
		{
			System.err.println("IOException occurred: " + ex.toString());
		}
	}

	private static void queryIndexedDocumentsByFieldAndId()
	{
		try
		{
			final String urlString = "http://localhost:8983/solr/techproducts";
			final HttpSolrClient solrClient = new HttpSolrClient.Builder(urlString).build();
			solrClient.setParser(new XMLResponseParser());
	
			final SolrQuery query = new SolrQuery();
			query.set("q", "price:599.99");
			final QueryResponse response = solrClient.query(query);
			 
			final SolrDocumentList docList = response.getResults();
			System.out.println("#Founds = " + docList.getNumFound());
			 
			for (SolrDocument doc : docList)
			{
				System.out.println("Id = " + doc.getFieldValue("id"));
				System.out.println("Price = " + doc.getFieldValue("price"));
			}

			//
			final SolrDocument doc = solrClient.getById("888");
			System.out.println("Name = " + doc.getFieldValue("name"));
			System.out.println("Price = " + doc.getFieldValue("price"));
		}
		catch (SolrServerException ex)
		{
			System.err.println("SolrServerException occurred: " + ex.toString());
		}
		catch (java.io.IOException ex)
		{
			System.err.println("IOException occurred: " + ex.toString());
		}
	}
	
	private static void deleteDocuments()
	{
		try
		{
			final String urlString = "http://localhost:8983/solr/techproducts";
			final HttpSolrClient solrClient = new HttpSolrClient.Builder(urlString).build();
			solrClient.setParser(new XMLResponseParser());
	
			{
				solrClient.deleteById("123456");
				solrClient.commit();
				
				final SolrQuery query = new SolrQuery();
				query.set("q", "id:123456");
				final QueryResponse response = solrClient.query(query);
				final SolrDocumentList docList = response.getResults();
				System.out.println("#Founds = " + docList.getNumFound());
			}

			{
				solrClient.deleteByQuery("name:Apple iPhone 6s");
				solrClient.commit();
				
				final SolrQuery query = new SolrQuery();
				query.set("q", "id:888");
				final QueryResponse response = solrClient.query(query);
				final SolrDocumentList docList = response.getResults();
				System.out.println("#Founds = " + docList.getNumFound());
			}
		}
		catch (SolrServerException ex)
		{
			System.err.println("SolrServerException occurred: " + ex.toString());
		}
		catch (java.io.IOException ex)
		{
			System.err.println("IOException occurred: " + ex.toString());
		}
	}
	
}

class ProductBean
{
	public ProductBean(String id, String name, String price)
	{
		super();
		this.id = id;
		this.name = name;
		this.price = price;
	}
/* 
    @Field("id")
	public String getId()
    {
        return id;
    }
    @Field("id")
    public void setId(String id)
    {
        this.id = id;
    }
 
    @Field("name")
    public String getName()
    {
        return name;
    }
    @Field("name")
    public void setName(String name)
    {
        this.name = name;
    }
 
    @Field("price")
    public String getPrice()
    {
        return price;
    }
    @Field("price")
    public void setPrice(String price)
    {
        this.price = price;
    }
*/
    @Field public String id;
    @Field public String name;
    @Field public String price;
}
