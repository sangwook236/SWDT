#include <arff_instance.h>



ArffInstance::ArffInstance(): m_size(0), m_data() {
}

ArffInstance::~ArffInstance() {
    std::vector<ArffValue*>::iterator itr;
    for(itr=m_data.begin();itr!=m_data.end();++itr) {
        delete *itr;
    }
}

int32 ArffInstance::size() const {
    return m_size;
}

void ArffInstance::add(ArffValue* val) {
    m_data.push_back(val);
    ++m_size;
}

ArffValue* ArffInstance::get(int idx) const {
    if((idx < 0) || (idx >= m_size)) {
        THROW("ArffInstance::get Index out of bounds! idx=%d size=%d",
              idx, m_size);
    }
    return m_data[idx];
}
