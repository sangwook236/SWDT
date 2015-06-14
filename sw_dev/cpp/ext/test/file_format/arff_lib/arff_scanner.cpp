#include <arff_scanner.h>


const char ArffScanner::NEWLINE = '\n';


ArffScanner::ArffScanner(const std::string& _file): m_file(_file),
                                            m_line((int64)0),
                                            m_col((int64)0),
                                            m_char(NEWLINE),
                                            m_prev_char((char)0),
                                            m_fp(NULL) {
    m_fp = fopen(m_file.c_str(), "r");
    if(m_fp == NULL) {
        THROW("ArffScanner: failed to open the file '%s'!", m_file.c_str());
    }
}

ArffScanner::~ArffScanner() {
    if(m_fp != NULL) {
        fclose(m_fp);
        m_fp = NULL;
    }
}

bool ArffScanner::is_newline(char c) const {
    return (c == NEWLINE);
}

char ArffScanner::next() {
    if(eof()) {
        return (char)-1;
    }
    if(is_newline(m_char)) {
        ++m_line;
        m_col = 0;
    }
    ++m_col;
    m_prev_char = m_char;
    if(fread(&m_char, sizeof(m_char), 1, m_fp) != sizeof(m_char)) {
        m_char = (char)-1;  // you would have reached end-of-file?
    }
    return m_char;
}

char ArffScanner::current() const {
    return m_char;
}

char ArffScanner::previous() const {
    return m_prev_char;
}

int64 ArffScanner::line() const {
    return m_line;
}

int64 ArffScanner::column() const {
    return m_col;
}

bool ArffScanner::eof() const {
    return feof(m_fp);
}

std::string ArffScanner::err_msg(const std::string& msg) const {
    std::string err = m_file + ":" + num2str<int64>(m_line);
    err += ":" + num2str<int64>(m_col) + " " + msg;
    return err;
}
