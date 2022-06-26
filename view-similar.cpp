/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 */

#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char** argv){

    std::string line;
    cv::namedWindow("view-similar", cv::WINDOW_GUI_EXPANDED );

    while(std::getline(std::cin, line)){
        // split line
        std::stringstream line_stream(line);
        std::vector< std::string > files;
        std::string file;
        std::vector< cv::Mat > images;
        int current_image = 0;

        while(std::getline(line_stream, file, '\t')){
            files.push_back(file);
            images.push_back(cv::imread(file, 1));
        }

        if(files.size() == 0){
            continue;
        }

        int key = 1;
        while(key){
            std::stringstream status_stream;
            status_stream << current_image+1 << "/" << files.size() << " " << files.at(current_image) << " (" << images.at(current_image).cols << "x" << images.at(current_image).rows << ")";
            
            if(key == 27 || key == 113){ // esc or q: quit
                return 0;
            }else if(key == 81){ // left arrow
                current_image = current_image > 0 ? current_image - 1 : 0;
            }else if(key == 83){ // right arrow
                current_image = current_image < files.size() - 1 ? current_image + 1 : current_image;
            }else if(key == 84){ // down arrow
                break;
            }else if(key == 32){ // space
                std::cout << files.at(current_image) << "\n";
            }

            // show image
            cv::imshow("view-similar", images.at(current_image));
            cv::displayStatusBar("view-similar", status_stream.str());

            key = cv::waitKey(0);
        }
    }
    return 0;
}
