/*
 * end_to_end_recognition
 *
 * This program reproduce the results of the Baseline (CV3 + Tess) in the
 * ICDAR2015 Robust Reading Competition.
 *
 * Originates from the OpenCV 3.0 demo program for End-to-end Scene Text Detection and Recognition:
 * Makes use of the Tesseract OCR API with the Extremal Region Filter algorithm.
 *
 * Created on: May 3, 2015
 *     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
 */

#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::text;

//Calculate approximate edit distance between two words
size_t edit_distance(const string& A, const string& B);
//Return minimum value of a triplet
size_t min(size_t x, size_t y, size_t z);
//Evaluate if a word is a repetitive pattern of low confidence characters
bool   isRepetitive(const string& s);
//Sort strings by lenght
bool   sort_by_lenght(const string &a, const string &b);
//Draw ER's in an image via floodFill
void   er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);


/* 
    Perform text detection and recognition and output results to stdout.
    The program accepts two command-line arguments: <input_image> [<input_txt_lexicon>]

    [<input_txt_lexicon>] is a txt file with lexicon words in the same format as the grountruth
                          of the ICDAR2015 Robust Reading Competition. It is optional and when not
                          provided the program returns the recognition results provided by Tesseract.

    Provided XML files with trained models for the ERFilter class must be in the execution path.
                          trained_classifierNM1.xml and trained_classifierNM2.xml

*/
int main(int argc, char* argv[])
{

    Mat image;
    if(argc>1)
    {
        image  = imread(argv[1]);
    }
    else
    {
        cout << "    Usage: " << argv[0] << " <input_image> [<input_txt_lexicon>]" << endl;
        return(0);
    }

    // Load lexicon if provided
    vector<string> lex;
    if(argc>2)
    {
        ifstream infile(argv[2]);
        string lex_word;
        while (infile >> lex_word)
        {
            lex.push_back(lex_word);
        }
    }

    /*Text Detection*/

    // Extract channels to be processed individually
    vector<Mat> channels;

    Mat grey;
    cvtColor(image,grey,COLOR_RGB2GRAY);


    // Extract channels to be processed individually
    // computeNMChannels(image, channels);
    // Notice here we are only using grey channel.
    channels.push_back(grey);
    channels.push_back(255-grey);

    // Create ERFilter objects with the 1st and 2nd stage default classifiers
    Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"),8,0.00015f,0.13f,0.2f,true,0.1f);
    Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"),0.5);

    vector<vector<ERStat> > regions(channels.size());
    // Apply the default cascade classifier to each independent channel
    for (int c=0; c<(int)channels.size(); c++)
    {
        er_filter1->run(channels[c], regions[c]);
        er_filter2->run(channels[c], regions[c]);
    }

    vector<string> final_words;
    vector<Rect>   final_boxes;
    vector<float>  final_confs;


    // Detect character groups
    vector< vector<Vec2i> > nm_region_groups;
    vector<Rect> nm_boxes;
    erGrouping(image, channels, regions, nm_region_groups, nm_boxes,ERGROUPING_ORIENTATION_HORIZ);



    /*Text Recognition (OCR)*/

    Ptr<OCRTesseract> ocr = OCRTesseract::create();
    string output;


    for (int i=0; i<(int)nm_boxes.size(); i++)
    {

        Mat group_img = Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
        er_draw(channels, regions, nm_region_groups[i], group_img);
        group_img(nm_boxes[i]).copyTo(group_img);
        copyMakeBorder(group_img,group_img,15,15,15,15,BORDER_CONSTANT,Scalar(0));

        vector<Rect>   boxes;
        vector<string> words;
        vector<float>  confidences;
        ocr->run(group_img, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

        output.erase(remove(output.begin(), output.end(), '\n'), output.end());
        //cout << "OCR output = \"" << output << "\" lenght = " << output.size() << endl;
        if (output.size() < 3)
            continue;

        for (int j=0; j<(int)boxes.size(); j++)
        {
            boxes[j].x += nm_boxes[i].x-15;
            boxes[j].y += nm_boxes[i].y-15;

            /* Filter words with low confidence, or less than 3 chars, or noisy recognitions */
            // Threshold values have been set using the ICDAR train set.
            //cout << "  word = " << words[j] << "\t confidence = " << confidences[j] << endl;
            if ((words[j].size() < 2) || (confidences[j] < 51) ||
                    ((words[j].size()==2) && (words[j][0] == words[j][1])) ||
                    ((words[j].size()< 4) && (confidences[j] < 60)) ||
                    isRepetitive(words[j]))
                continue;

            /* Increase confidence of predicted words matching a word in the lexicon */
            if (lex.size() > 0)
            {
                std::transform(words[j].begin(), words[j].end(), words[j].begin(), ::toupper);
                if (find(lex.begin(), lex.end(), words[j]) == lex.end())
                    confidences[j] = 200;
            }

            final_words.push_back(words[j]);
            final_boxes.push_back(boxes[j]);
            final_confs.push_back(confidences[j]);
        }

    }


    /* Non Maximal Suppression using OCR confidence */
    float thr = 0.5;

    for (int i=0; i<final_words.size(); )
    {
        int to_delete = -1;
        for (int j=i+1; j<final_words.size(); )
        {
            to_delete = -1;
            Rect intersection = final_boxes[i] & final_boxes[j];
            float IoU = (float)intersection.area() / (final_boxes[i].area() + final_boxes[j].area() - intersection.area());
            if ((IoU > thr) || (intersection.area() > 0.8*final_boxes[i].area()) || (intersection.area() > 0.8*final_boxes[j].area()))
            {
                // if regions overlap more than thr delete the one with lower confidence
                to_delete = (final_confs[i] < final_confs[j]) ? i : j;

                if (to_delete == j )
                {
                    final_words.erase(final_words.begin()+j);
                    final_boxes.erase(final_boxes.begin()+j);
                    final_confs.erase(final_confs.begin()+j);
                    continue;
                } else {
                    break;
                }
            }
            j++;
        }
        if (to_delete == i )
        {
            final_words.erase(final_words.begin()+i);
            final_boxes.erase(final_boxes.begin()+i);
            final_confs.erase(final_confs.begin()+i);
            continue;
        }
        i++;
    }

    /* Predicted words which are not in the lexicon are filtered
       or changed to match one (when edit distance ratio < 0.34)*/
    float max_edit_distance_ratio = 0.34;
    for (int j=0; j<final_boxes.size(); j++)
    {

        if (lex.size() > 0)
        {
            std::transform(final_words[j].begin(), final_words[j].end(), final_words[j].begin(), ::toupper);
            if (find(lex.begin(), lex.end(), final_words[j]) == lex.end())
            {
                int best_match = -1;
                int best_dist  = final_words[j].size();
                for (int l=0; l<lex.size(); l++)
                {
                    int dist = edit_distance(lex[l],final_words[j]);
                    if (dist < best_dist)
                    {
                        best_match = l;
                        best_dist = dist;
                    }
                }
                if (best_dist/final_words[j].size() < max_edit_distance_ratio)
                    final_words[j] = lex[best_match];
                else
                    continue;
            }
        }

        // Output final recognition in csv format compatible with the ICDAR Robust Reading Competition
        cout << final_boxes[j].tl().x << ","
             << final_boxes[j].tl().y << ","
             << min(final_boxes[j].br().x,image.cols-2)
             << "," << final_boxes[j].tl().y << ","
             << min(final_boxes[j].br().x,image.cols-2) << ","
             << min(final_boxes[j].br().y,image.rows-2) << ","
             << final_boxes[j].tl().x << ","
             << min(final_boxes[j].br().y,image.rows-2) << ","
             << final_words[j] << endl ;
    }

    return 0;
}



//Return minimum value of a triplet
size_t min(size_t x, size_t y, size_t z)
{
    return x < y ? min(x,z) : min(y,z);
}

//Calculate approximate edit distance between two words
size_t edit_distance(const string& A, const string& B)
{
    size_t NA = A.size();
    size_t NB = B.size();

    vector< vector<size_t> > M(NA + 1, vector<size_t>(NB + 1));

    for (size_t a = 0; a <= NA; ++a)
        M[a][0] = a;

    for (size_t b = 0; b <= NB; ++b)
        M[0][b] = b;

    for (size_t a = 1; a <= NA; ++a)
        for (size_t b = 1; b <= NB; ++b)
        {
            size_t x = M[a-1][b] + 1;
            size_t y = M[a][b-1] + 1;
            size_t z = M[a-1][b-1] + (A[a-1] == B[b-1] ? 0 : 1);
            M[a][b] = min(x,y,z);
        }

    return M[A.size()][B.size()];
}

//Evaluate if a word is a repetitive pattern of low confidence characters
bool isRepetitive(const string& s)
{
    int count = 0;
    for (int i=0; i<(int)s.size(); i++)
    {
        if ((s[i] == 'i') ||
                (s[i] == 'l') ||
                (s[i] == 'I'))
            count++;
    }
    if (count > ((int)s.size()+1)/2)
    {
        return true;
    }
    return false;
}

//Draw ER's in an image via floodFill
void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
{
    for (int r=0; r<(int)group.size(); r++)
    {
        ERStat er = regions[group[r][0]][group[r][1]];
        if (er.parent != NULL) // deprecate the root region
        {
            int newMaskVal = 255;
            int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
            floodFill(channels[group[r][0]],segmentation,Point(er.pixel%channels[group[r][0]].cols,er.pixel/channels[group[r][0]].cols),
                      Scalar(255),0,Scalar(er.level),Scalar(0),flags);
        }
    }
}

//Sort strings by lenght
bool sort_by_lenght(const string &a, const string &b){return (a.size()>b.size());}
