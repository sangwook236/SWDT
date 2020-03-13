package tika;

import java.io.IOException;
import java.io.InputStream;

import org.apache.tika.Tika;
import org.apache.tika.detect.DefaultDetector;
import org.apache.tika.detect.Detector;
import org.apache.tika.exception.TikaException;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.mime.MediaType;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.Parser;
import org.apache.tika.sax.BodyContentHandler;
import org.junit.Test;
import org.xml.sax.ContentHandler;
import org.xml.sax.SAXException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;
import static org.junit.matchers.JUnitMatchers.containsString;

public class Tika_Main {

	public static void run(final String[] args)
	{
		// REF [site] >> https://www.baeldung.com/apache-tika
		try {
			whenUsingDetector_thenDocumentTypeIsReturned();
			whenUsingParser_thenContentIsReturned();
			whenUsingParser_thenMetadataIsReturned();
		}
		catch (Exception ex) {
			System.out.println("Exception caught: " + ex);
		}
	}

	static String detectDocTypeUsingDetector(final InputStream stream) throws IOException {
		final Detector detector = new DefaultDetector();
		final Metadata metadata = new Metadata();

		final MediaType mediaType = detector.detect(stream, metadata);
		return mediaType.toString();
	}

	static String detectDocTypeUsingFacade(final InputStream stream) throws IOException {
		final Tika tika = new Tika();
		final String mediaType = tika.detect(stream);
		return mediaType;
	}

	static String extractContentUsingParser(final InputStream stream) throws IOException, SAXException, TikaException {
		final Parser parser = new AutoDetectParser();
		final ContentHandler handler = new BodyContentHandler();
		final Metadata metadata = new Metadata();
		final ParseContext context = new ParseContext();

		parser.parse(stream, handler, metadata, context);
		return handler.toString();
	}

	static String extractContentUsingFacade(final InputStream stream) throws IOException, TikaException {
		final Tika tika = new Tika();
		final String content = tika.parseToString(stream);
		return content;
	}

	static Metadata extractMetadatatUsingParser(final InputStream stream) throws IOException, SAXException, TikaException {
		final Parser parser = new AutoDetectParser();
		final ContentHandler handler = new BodyContentHandler();
		final Metadata metadata = new Metadata();
		final ParseContext context = new ParseContext();

		parser.parse(stream, handler, metadata, context);
		return metadata;
	}

	static Metadata extractMetadatatUsingFacade(final InputStream stream) throws IOException, TikaException {
		final Tika tika = new Tika();
		final Metadata metadata = new Metadata();

		tika.parse(stream, metadata);
		return metadata;
	}

	@Test
	static void whenUsingDetector_thenDocumentTypeIsReturned() throws IOException, ClassNotFoundException
	{
		final InputStream stream = Class.forName("Tika_Main").getClassLoader().getResourceAsStream("tika.txt");
		final String mediaType = Tika_Main.detectDocTypeUsingDetector(stream);

		assertEquals("application/pdf", mediaType);

		stream.close();
	}

	@Test
	static void whenUsingParser_thenContentIsReturned() throws IOException, SAXException, TikaException, ClassNotFoundException
	{
		final InputStream stream = Class.forName("Tika_Main").getClassLoader().getResourceAsStream("tika.docx");
		final String content = Tika_Main.extractContentUsingParser(stream);

		assertThat(content, containsString("Apache Tika - a content analysis toolkit"));
		assertThat(content, containsString("detects and extracts metadata and text"));

		stream.close();
	}

	@Test
	static void whenUsingParser_thenMetadataIsReturned() throws IOException, SAXException, TikaException, ClassNotFoundException
	{
		final InputStream stream = Class.forName("Tika_Main").getClassLoader().getResourceAsStream("tika.xlsx");
		final Metadata metadata = Tika_Main.extractMetadatatUsingParser(stream);

		assertEquals("org.apache.tika.parser.DefaultParser", metadata.get("X-Parsed-By"));
		assertEquals("Microsoft Office User", metadata.get("Author"));

		stream.close();
	}
}
