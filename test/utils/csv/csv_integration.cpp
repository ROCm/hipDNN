#include "csv_integration.hpp"

void write_to_csv(std::string filename, std::string str, std::string testname,
                  double avg_time, std::string str_ip_size, 
                  std::string str_k_size, std::string str_op_size){
	std::fstream fs2;
	fs2.open(filename, std::ios::in);
	std::fstream fs;
	fs.open(filename, std::ios_base::app);	
   if (!fs2.is_open()) {
        
        fs << "Test_name"
         << ","
         << "Output" 
         << ","
         << "Average Excecution Time (milliseconds)"
         << ","
         << "Input size"
         << ","
         << "kernel size"
         << ","
         << "output size"
         << std::endl;
   }
  fs << testname << ",";
  fs << str << ",";
  fs << avg_time << ",";
  fs << str_ip_size << ",";
  fs << str_k_size << ",";
  fs << str_op_size << std::endl;
  fs.close();
}
