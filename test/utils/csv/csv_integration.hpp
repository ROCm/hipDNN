#ifndef CSV_INTEGRATION_HPP
#define CSV_INTEGRATION_HPP

#include <string>
#include <iostream>
#include <fstream>

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

void write_to_csv(std::string filename, std::string str, std::string testname) {
    std::fstream fs;

  fs.open(filename, std::ios_base::app);

  fs << "Test_name"
     << ","
     << "Output" << std::endl;

  fs << testname << ",";

  fs << str << std::endl;

  fs.close();
}

#endif // CSV_INTEGRATION_HPP
