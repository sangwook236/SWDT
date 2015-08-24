/** 
    This file is part of the implementation of the people detection and pose estimation model as described in the paper:
    
    M. Andriluka, S. Roth, B. Schiele. 
    Pictorial Structures Revisited: People Detection and Articulated Pose Estimation. 
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR'09), Miami, USA, June 2009

    Please cite the paper if you are using this code in your work.

    The code may be used free of charge for non-commercial and
    educational purposes, the only requirement is that this text is
    preserved within the derivative work. For any other purpose you
    must contact the authors for permission. This code may not be
    redistributed without permission from the authors.  

    Author: Micha Andriluka, 2009
	    andriluka@cs.tu-darmstadt.de
	    http://www.mis.informatik.tu-darmstadt.de/People/micha
*/

#ifndef _PROTOBUF_AUX_H_
#define _PROTOBUF_AUX_H_

#include <QString>
#include <fstream>
#include <iostream>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <libFilesystemAux/filesystem_aux.h>

/**
   load and save protocol buffer messages in text format
*/

template <typename PB_CLASS>
bool write_message_binary(QString qsFilename, PB_CLASS &msg)
{
  using namespace std;
  fstream out(qsFilename.toStdString().c_str(), ios::out | ios::trunc | ios::binary);
  bool bRes = msg.SerializeToOstream(&out);
  assert(bRes);
  return bRes;
}

template <typename PB_CLASS>
bool parse_message_binary(QString qsFilename, PB_CLASS &msg)
{
  using namespace std;
  assert(filesys::check_file(qsFilename));

  fstream in(qsFilename.toStdString().c_str(), ios::in | ios::binary);
  bool bRes = msg.ParseFromIstream(&in);
  assert(bRes);
  return bRes;
}

template <typename PB_CLASS>
void parse_message_from_text_file(QString qsFilename, PB_CLASS &msg)
{
  //--S [] 2013/01/10: Sang-Wook Lee
  //std::fstream fstr_in(qsFilename.toAscii().data(), std::ios::in);
  std::fstream fstr_in(qsFilename.toStdString().c_str(), std::ios::in);
  //--S [] 2013/01/10: Sang-Wook Lee
  google::protobuf::io::ZeroCopyInputStream *zc_in = new google::protobuf::io::IstreamInputStream(&fstr_in);

  if (!filesys::check_file(qsFilename)) {
    std::cout << "file not found: " << qsFilename.toStdString() << std::endl;
    assert(false);
  }

  bool bRes = google::protobuf::TextFormat::Parse(zc_in, &msg);  
  assert(bRes && "error while parsing protobuf file");

  delete zc_in;
}

template <typename PB_CLASS>
void print_message_to_text_file(QString qsFilename, const PB_CLASS &msg)
{
  //--S [] 2013/01/10: Sang-Wook Lee
  //std::fstream fstr_out(qsFilename.toAscii().data(), std::ios::out | std::ios::trunc);
  std::fstream fstr_out(qsFilename.toStdString().c_str(), std::ios::out | std::ios::trunc);
  //--E [] 2013/01/10
  google::protobuf::io::ZeroCopyOutputStream *zc_out = new google::protobuf::io::OstreamOutputStream(&fstr_out);

  bool bRes = google::protobuf::TextFormat::Print(msg, zc_out);
  assert(bRes && "error while saving protobuf file");

  delete zc_out;  
}


#endif
