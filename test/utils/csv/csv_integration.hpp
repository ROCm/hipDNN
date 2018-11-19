#ifndef CSV_INTEGRATION_HPP
#define CSV_INTEGRATION_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

template <typename dataType>
std::string convert_to_string(dataType *arr, int s_size) {
  std::ostringstream os;

  os << "\"[";

  for (int i = 0; i < s_size; i++) {
    os << arr[i];

    if (i < (s_size - 1))
      os << ",";
  }

  os << "]\"";

  std::string str(os.str());

  return str;
}

template <typename dataType>
std::string convert_to_string2(dataType *arr, int s_size) {
  std::ostringstream os;

  os << "[";

  for (int i = 0; i < s_size; i++) {
    os << arr[i];

    if (i < (s_size - 1))
      os << ",";
  }

  os << "]";

  std::string str(os.str());

  return str;
}

template <typename dataType>
std::string integration_dims_to_string(dataType *arr1, dataType *arr2, std::string a, std::string b) {
  std::ostringstream os;

  os << "\""<<a<<":[";

  for (int i = 0; i < 4; i++) {
    os << arr1[i];

    if (i < 3)
      os << ",";
  }

  os << "]"<< b <<":[";

  for (int i = 0; i < 4; i++) {
    os << arr2[i];

    if (i < 3)
      os << ",";
  }

  os << "]\"";

  std::string str(os.str());

  return str;
}

template <typename dataType>
std::string integration_dims_to_string2(dataType *arr1, dataType *arr2,dataType *arr3, dataType *arr4, std::string a, std::string b, std::string c, std::string d) {
  std::ostringstream os;

  os << "\""<<a<<":[";

  for (int i = 0; i < 4; i++) {
    os << arr1[i];

    if (i < 3)
      os << ",";
  }

  os << "]"<< b <<":[";

  for (int i = 0; i < 4; i++) {
    os << arr2[i];

    if (i < 3)
      os << ",";
  }
  
    os << "]"<< c <<":[";

  for (int i = 0; i < 4; i++) {
    os << arr3[i];

    if (i < 3)
      os << ",";
  }

  os << "]"<< d <<":[";

  for (int i = 0; i < 4; i++) {
    os << arr4[i];

    if (i < 3)
      os << ",";
  }

  os << "]\"";

  std::string str(os.str());

  return str;
}

template <typename dataType>
std::string integration_dims_to_string3(dataType *arr1, dataType *arr2,dataType *arr3, dataType *arr4, dataType *arr5, dataType *arr6, std::string a, std::string b, std::string c, std::string d, std::string e, std::string f) {
  std::ostringstream os;

  os << "\""<<a<<":[";

  for (int i = 0; i < 4; i++) {
    os << arr1[i];

    if (i < 3)
      os << ",";
  }

  os << "]"<< b <<":[";

  for (int i = 0; i < 4; i++) {
    os << arr2[i];

    if (i < 3)
      os << ",";
  }
  
    os << "]"<< c <<":[";

  for (int i = 0; i < 4; i++) {
    os << arr3[i];

    if (i < 3)
      os << ",";
  }

  os << "]"<< d <<":[";

  for (int i = 0; i < 4; i++) {
    os << arr4[i];

    if (i < 3)
      os << ",";
  }

  os << "]"<< e <<":[";

  for (int i = 0; i < 4; i++) {
    os << arr5[i];

    if (i < 3)
      os << ",";
  }

  os << "]"<< f <<":[";

  for (int i = 0; i < 4; i++) {
    os << arr6[i];

    if (i < 3)
      os << ",";
  }


  os << "]\"";

  std::string str(os.str());

  return str;
}

void write_to_csv(std::string filename, std::string str, std::string testname,
                  float avg_time, std::string str_ip_size, 
                  std::string str_k_size, std::string str_op_size);

#endif // CSV_INTEGRATION_HPP
