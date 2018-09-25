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

void write_to_csv(std::string filename, std::string str, std::string testname,
                  float avg_time, std::string str_ip_size, 
                  std::string str_k_size, std::string str_op_size);

#endif // CSV_INTEGRATION_HPP
