#include "csv_integration.hpp"

void write_to_csv(std::string filename, std::string str, std::string testname,
                  float avg_time, std::string str_ip_size,
                  std::string str_k_size, std::string str_op_size){

  std::fstream fs2;
  fs2.open(filename, std::ios::in);

  std::fstream fs;
  fs.open(filename, std::ios_base::app);

   if (!fs2.is_open()) {

        fs << "Test_name"
           << ","
           << "Average Excecution Time (microseconds)"
           << ","
           << "Input size"
           << ","
           << "kernel size"
           << ","
           << "output size"
           << ","
           << "Output"
           << std::endl;
   }

  fs << testname << ",";
  fs << avg_time << ",";
  fs << str_ip_size << ",";
  fs << str_k_size << ",";
  fs << str_op_size << ",";
  fs << str << std::endl;

  fs.close();

}

void dump_result_csv(std::string filename, std::string testname, float* output,
                     int n){

  if (mkdir("./results_csv", 0777) == -1)
        std::cerr << strerror(errno) << std::endl;

  else
        std::cout << "Directory created \n";

  std::string file_path = "./results_csv/" + filename;

  std::fstream fs2;
  fs2.open(file_path, std::ios::in);

  std::fstream fs;
  fs.open(file_path, std::ios_base::app);

  if (!fs2.is_open()) {

      fs << "Test_name"
         << ","
         << "Output"
         << std::endl;
   }

  fs << testname << ",";

  for (int i=0; i<n; i++)

      fs << *(output+i) << ",";

  fs << std::endl;

  fs.close();

}