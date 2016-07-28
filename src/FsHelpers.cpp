#include "FsHelpers.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>


std::string FsHelpers::CombinePath(
    const std::string& path1,
    const std::string& path2)
{
    boost::filesystem::path p1(path1);
    boost::filesystem::path p2(path2);

    return (p1 / p2).string();
}


std::string FsHelpers::CombinePath(
    const std::string& path1,
    const std::string& path2,
    const std::string& path3)
{
    boost::filesystem::path p1(path1);
    boost::filesystem::path p2(path2);
    boost::filesystem::path p3(path3);

    return (p1 / p2 / p3).string();
}


std::string FsHelpers::CombinePath(
    const std::string& path1,
    const std::string& path2,
    const std::string& path3,
    const std::string& path4)
{
    boost::filesystem::path p1(path1);
    boost::filesystem::path p2(path2);
    boost::filesystem::path p3(path3);
    boost::filesystem::path p4(path4);

    return (p1 / p2 / p3 / p4).string();
}


std::string FsHelpers::CombinePath(const std::vector<std::string>& paths)
{
    boost::filesystem::path p;

    for (int i = 0; i < paths.size(); ++i)
    {
        p /= boost::filesystem::path(paths[i]);
    }

    return p.string();
}


std::string FsHelpers::GetFileName(const std::string& path)
{
    boost::filesystem::path p(path);

    return p.filename().string();
}


std::string FsHelpers::GetExtension(const std::string& path)
{
    boost::filesystem::path p(path);

    return p.extension().string();
}


std::string FsHelpers::GetFileNameWithoutExtension(const std::string& path)
{
    boost::filesystem::path p(path);

    return p.stem().string();
}


std::string FsHelpers::GetParentPath(const std::string& path)
{
    boost::filesystem::path p(path);

    return p.parent_path().string();
}


std::string FsHelpers::GetParentName(const std::string& path)
{
    boost::filesystem::path p(path);

    return p.parent_path().filename().string();
}


bool FsHelpers::MakeDirectory(const std::string& path)
{
    boost::filesystem::path p(path);

    return create_directory(p);
}


bool FsHelpers::MakeDirectories(const std::string& path)
{
    boost::filesystem::path p(path);

    return create_directories(p);
}


void FsHelpers::GetDirectories(
    const std::string& path,
    std::vector<std::string>& out_paths)
{
    out_paths.clear();

    boost::filesystem::path p(path);

    // non-recursively search
    boost::filesystem::directory_iterator endIter;
    for (boost::filesystem::directory_iterator iter(p);
        iter != endIter; ++iter)
    {
        if (boost::filesystem::is_directory(iter->path()))
        {
            out_paths.push_back(iter->path().string());
        }
    }
}


void FsHelpers::GetFilesMatchRegex(
    const std::string& path,
    std::vector<std::string>& out_paths,
    const std::string& regexFilter,
    bool recursively)
{
    //throw new std::exception("not implemented yet");
    return;

    out_paths.clear();

    boost::filesystem::path p(path);
    boost::regex filter(regexFilter);

    if (recursively == false)
    {
        // non-recursively search
        boost::filesystem::directory_iterator endIter;
        for (boost::filesystem::directory_iterator iter(p);
             iter != endIter; ++iter)
        {
            if (boost::filesystem::is_regular_file(iter->path()))
            {
                boost::smatch dummy;
                std::string filenameStr = iter->path().filename().string();
                if (!boost::regex_match(filenameStr, dummy, filter))
                    continue;
                out_paths.push_back(iter->path().string());
            }
        }
    }
    else
    {
        // recursively search
    }
}

void FsHelpers::GetFilesHasExtension(
    const std::string& path,
    std::vector<std::string>& out_paths,
    const std::string& extension,
    bool recursively)
{
    //out_paths.clear();

    std::string ext = extension;
    // add dot
    if (ext[0] != '.')
    {
        ext = "." + ext;
    }

    boost::filesystem::path p(path);

    if (recursively == false)
    {
        // non-recursively search
        boost::filesystem::directory_iterator endIter;
        for (boost::filesystem::directory_iterator iter(p);
             iter != endIter; ++iter)
        {
            if (boost::filesystem::is_regular_file(iter->path()) &&
                iter->path().extension().string() == ext)
            {
                out_paths.push_back(iter->path().string());
            }
        }
    }
    else
    {
        // recursively search
        boost::filesystem::recursive_directory_iterator endIter;
        for (boost::filesystem::recursive_directory_iterator iter(p);
             iter != endIter; ++iter)
        {
            if (boost::filesystem::is_regular_file(iter->path()) &&
                iter->path().extension().string() == ext)
            {
                out_paths.push_back(iter->path().string());
            }
        }
    }
}


bool FsHelpers::Exists(const std::string& path)
{
    boost::filesystem::path p(path);

    return boost::filesystem::exists(p);
}


bool FsHelpers::ReadAllLines(const std::string& filePath, std::vector<std::string>& out_allLines)
{
    std::ifstream fileStream(filePath.c_str());
    if (!fileStream.is_open())
    {
        std::cerr << "Failed to open file at: " << filePath << std::endl;
        return false;
    }

    std::string textLine;
    while (!fileStream.eof())
    {
        std::getline(fileStream, textLine);
        out_allLines.push_back(textLine);
    }

    return true;
}

