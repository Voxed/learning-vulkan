/**
 * main.cpp
 * 
 * Playthrough of the vulkan tutorial https://vulkan-tutorial.com/
 */
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <cstdint>
#include <fstream>

const int MAX_FRAMES_IN_FLIGHT = 2;

static std::vector<char> readFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value();
    }
};

struct SwapChainSupportDetails
{
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

const int WIDTH = 800;
const int HEIGHT = 600;

/**
 * The playthrough.
 */
class Playthrough
{
public:
    /**
     * Start of the playthrough.
     */
    void Run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow *window;
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::SurfaceKHR surface;
    vk::SwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::ImageView> swapChainImageViews;
    vk::PipelineLayout pipelineLayout;
    vk::RenderPass renderPass;
    vk::Pipeline graphicsPipeline;
    std::vector<vk::Framebuffer> swapChainFramebuffers;
    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;
    vk::Semaphore imageAvailableSemaphore;
    vk::Semaphore renderFinishedSemaphore;
    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;
    size_t currentFrame = 0;

    /**
     * Open window using GLFW for cross platform support.
     */
    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Playthrough", nullptr, nullptr);
    }

    /**
     * Initialize the vulkan instance and devices.
     */
    void initVulkan()
    {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
    }

    /**
     * Create the vulkan instance with required extensions and layers.
     */
    void createInstance()
    {
        // Initialize application info.
        vk::ApplicationInfo appInfo =
            vk::ApplicationInfo()
                .setPApplicationName("Playthrough")
                .setPEngineName("No Engine")
                .setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
                .setApiVersion(VK_API_VERSION_1_2);

        // Initialize instance create info.
        vk::InstanceCreateInfo createInfo =
            vk::InstanceCreateInfo()
                .setPApplicationInfo(&appInfo)
                .setEnabledLayerCount(0);

        // Get required extensions.
        uint32_t requiredExtensionsCount;
        const char **pRequiredExtensions = glfwGetRequiredInstanceExtensions(&requiredExtensionsCount);
        std::vector<const char *> requiredExtensions(pRequiredExtensions, pRequiredExtensions + requiredExtensionsCount);

        // Set required extensions.
        createInfo.setPEnabledExtensionNames(requiredExtensions);

        // Set validation layers.
        std::vector<const char *> requiredValidationLayers{
            "VK_LAYER_KHRONOS_validation"};
        createInfo.setPEnabledLayerNames(requiredValidationLayers);

        // Create the instance.
        instance = vk::createInstance(createInfo);
    }

    void createSurface()
    {
        VkSurfaceKHR cSurface;
        glfwCreateWindowSurface(instance, window, nullptr, &cSurface);
        surface = cSurface;
    }

    /**
     * Pick suitable graphics device.
     */
    void pickPhysicalDevice()
    {
        for (const auto &device : instance.enumeratePhysicalDevices())
        {
            if (isDeviceSuitable(device))
            {
                physicalDevice = device;
                break;
            }
        }
    }

    /**
     * Check device suitability.
     */
    bool isDeviceSuitable(vk::PhysicalDevice device)
    {
        // Todo query swapchain support.

        QueueFamilyIndices indices = findQueueFamilies(device);

        return indices.isComplete();
    }

    /**
     * Query required queue families from physical device.
     */
    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device)
    {
        QueueFamilyIndices indices;

        unsigned int i = 0;
        for (const auto &queueProperties : device.getQueueFamilyProperties())
        {
            if (queueProperties.queueFlags & vk::QueueFlagBits::eGraphics)
            {
                indices.graphicsFamily = i;
            }

            if (device.getSurfaceSupportKHR(i, surface))
            {
                indices.presentFamily = i;
            }

            i++;
        }

        return indices;
    }

    /**
     * Create logical device with one graphics queue.
     */
    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

        float queuePriority = 1.0f;
        for (const auto &queue : {indices.graphicsFamily.value(), indices.presentFamily.value()})
        {
            vk::DeviceQueueCreateInfo queueCreateInfo =
                vk::DeviceQueueCreateInfo()
                    .setQueueCount(1)
                    .setQueuePriorities(queuePriority)
                    .setQueueFamilyIndex(queue);
            queueCreateInfos.push_back(queueCreateInfo);
        }

        vk::PhysicalDeviceFeatures deviceFeatures;

        std::vector<const char *> extensions{
            VK_KHR_SWAPCHAIN_EXTENSION_NAME};

        vk::DeviceCreateInfo createInfo =
            vk::DeviceCreateInfo()
                .setQueueCreateInfos(queueCreateInfos)
                .setPEnabledFeatures(&deviceFeatures)
                .setPEnabledExtensionNames(extensions);

        device = physicalDevice.createDevice(createInfo);
        graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
        presentQueue = device.getQueue(indices.presentFamily.value(), 0);
    }

    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR createInfo =
            vk::SwapchainCreateInfoKHR()
                .setSurface(surface)
                .setImageFormat(surfaceFormat.format)
                .setImageColorSpace(surfaceFormat.colorSpace)
                .setImageExtent(extent)
                .setImageArrayLayers(1)
                .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
                .setMinImageCount(imageCount);

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        std::vector<uint32_t> queueFamilyIndices = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        if (indices.graphicsFamily != indices.presentFamily)
        {
            createInfo
                .setImageSharingMode(vk::SharingMode::eConcurrent)
                .setQueueFamilyIndices(queueFamilyIndices);
        }
        else
        {
            createInfo.setImageSharingMode(vk::SharingMode::eExclusive);
        }

        createInfo.setPreTransform(swapChainSupport.capabilities.currentTransform);
        createInfo.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);
        createInfo.setPresentMode(presentMode);
        createInfo.setClipped(true);
        createInfo.setOldSwapchain(nullptr);

        swapChain = device.createSwapchainKHR(createInfo);
        swapChainImages = device.getSwapchainImagesKHR(swapChain);
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device)
    {
        SwapChainSupportDetails details;

        details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
        details.formats = device.getSurfaceFormatsKHR(surface);
        details.presentModes = device.getSurfacePresentModesKHR(surface);

        return details;
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
    {
        for (const auto &format : availableFormats)
        {
            if (format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            {
                return format;
            }
        }
        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> availablePresentModes)
    {
        for (const auto &availablePresentMode : availablePresentModes)
        {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox)
            {
                return availablePresentMode;
            }
        }

        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities)
    {
        if (capabilities.currentExtent.width != UINT32_MAX)
        {
            return capabilities.currentExtent;
        }
        else
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            vk::Extent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    void createImageViews()
    {
        for (const auto &image : swapChainImages)
        {
            vk::ImageViewCreateInfo createInfo =
                vk::ImageViewCreateInfo()
                    .setImage(image)
                    .setViewType(vk::ImageViewType::e2D)
                    .setFormat(swapChainImageFormat)
                    .setComponents(vk::ComponentMapping())
                    .setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
            vk::ImageView imageView = device.createImageView(createInfo);
            swapChainImageViews.push_back(imageView);
        }
    }

    void createRenderPass()
    {
        vk::AttachmentDescription colorAttachment =
            vk::AttachmentDescription()
                .setSamples(vk::SampleCountFlagBits::e1)
                .setFormat(swapChainImageFormat)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setStencilLoadOp(vk::AttachmentLoadOp::eClear)
                .setStencilStoreOp(vk::AttachmentStoreOp::eStore)
                .setInitialLayout(vk::ImageLayout::eUndefined)
                .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

        vk::AttachmentReference colorAttachmentRef =
            vk::AttachmentReference()
                .setAttachment(0)
                .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

        vk::SubpassDescription subpass =
            vk::SubpassDescription()
                .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                .setColorAttachments(colorAttachmentRef);

        vk::SubpassDependency dependency =
            vk::SubpassDependency()
                .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                .setDstSubpass(0)
                .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                .setSrcAccessMask(vk::AccessFlags())
                .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);

        vk::RenderPassCreateInfo createInfo =
            vk::RenderPassCreateInfo()
                .setAttachments(colorAttachment)
                .setSubpasses(subpass)
                .setDependencies(dependency);

        renderPass = device.createRenderPass(createInfo);
    }

    void createGraphicsPipeline()
    {
        auto vertShaderCode = readFile("vert.spv");
        auto fragShaderCode = readFile("frag.spv");

        vk::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        vk::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo =
            vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eVertex)
                .setModule(vertShaderModule)
                .setPName("main");

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo =
            vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eFragment)
                .setModule(fragShaderModule)
                .setPName("main");

        std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
            vertShaderStageInfo,
            fragShaderStageInfo};

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly =
            vk::PipelineInputAssemblyStateCreateInfo()
                .setTopology(vk::PrimitiveTopology::eTriangleList)
                .setPrimitiveRestartEnable(false);

        vk::Viewport viewport(0.0f, 0.0f, (float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f);

        vk::Rect2D scissor =
            vk::Rect2D()
                .setOffset({0, 0})
                .setExtent(swapChainExtent);

        vk::PipelineViewportStateCreateInfo viewportState =
            vk::PipelineViewportStateCreateInfo()
                .setViewports(viewport)
                .setScissors(scissor);

        vk::PipelineRasterizationStateCreateInfo rasterizer =
            vk::PipelineRasterizationStateCreateInfo()
                .setDepthClampEnable(false)
                .setRasterizerDiscardEnable(false)
                .setPolygonMode(vk::PolygonMode::eFill)
                .setLineWidth(1.0f)
                .setCullMode(vk::CullModeFlagBits::eBack)
                .setFrontFace(vk::FrontFace::eClockwise)
                .setDepthBiasEnable(false);

        vk::PipelineMultisampleStateCreateInfo multisampling =
            vk::PipelineMultisampleStateCreateInfo()
                .setSampleShadingEnable(false)
                .setRasterizationSamples(vk::SampleCountFlagBits::e1);

        vk::PipelineColorBlendAttachmentState colorBlendAttachment =
            vk::PipelineColorBlendAttachmentState()
                .setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
                .setBlendEnable(false);

        vk::PipelineColorBlendStateCreateInfo colorBlending =
            vk::PipelineColorBlendStateCreateInfo()
                .setLogicOpEnable(false)
                .setAttachments(colorBlendAttachment);

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo =
            vk::PipelineLayoutCreateInfo();

        pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo =
            vk::GraphicsPipelineCreateInfo()
                .setStages(shaderStages)
                .setPVertexInputState(&vertexInputInfo)
                .setPInputAssemblyState(&inputAssembly)
                .setPViewportState(&viewportState)
                .setPRasterizationState(&rasterizer)
                .setPMultisampleState(&multisampling)
                .setPColorBlendState(&colorBlending)
                .setLayout(pipelineLayout)
                .setRenderPass(renderPass)
                .setSubpass(0);

        graphicsPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo).value;

        device.destroyShaderModule(vertShaderModule);
        device.destroyShaderModule(fragShaderModule);
    }

    vk::ShaderModule createShaderModule(const std::vector<char> &code)
    {
        vk::ShaderModuleCreateInfo createInfo =
            vk::ShaderModuleCreateInfo()
                .setCodeSize(code.size())
                .setPCode(reinterpret_cast<const uint32_t *>(code.data()));

        return device.createShaderModule(createInfo);
    }

    void createFramebuffers()
    {
        for (auto &imageView : swapChainImageViews)
        {
            vk::FramebufferCreateInfo createInfo =
                vk::FramebufferCreateInfo()
                    .setRenderPass(renderPass)
                    .setAttachments(imageView)
                    .setWidth(swapChainExtent.width)
                    .setHeight(swapChainExtent.height)
                    .setLayers(1);

            swapChainFramebuffers.push_back(device.createFramebuffer(createInfo));
        }
    }

    void createCommandPool()
    {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        vk::CommandPoolCreateInfo poolInfo =
            vk::CommandPoolCreateInfo()
                .setQueueFamilyIndex(queueFamilyIndices.graphicsFamily.value());

        commandPool = device.createCommandPool(poolInfo);
    }

    void createCommandBuffers()
    {
        vk::CommandBufferAllocateInfo allocInfo =
            vk::CommandBufferAllocateInfo()
                .setCommandPool(commandPool)
                .setLevel(vk::CommandBufferLevel::ePrimary)
                .setCommandBufferCount(swapChainFramebuffers.size());

        commandBuffers = device.allocateCommandBuffers(allocInfo);

        int i = 0;
        for (const auto &commandBuffer : commandBuffers)
        {
            vk::CommandBufferBeginInfo beginInfo;

            commandBuffer.begin(beginInfo);

            vk::RenderPassBeginInfo renderPassInfo =
                vk::RenderPassBeginInfo()
                    .setRenderPass(renderPass)
                    .setFramebuffer(swapChainFramebuffers[i])
                    .setRenderArea(vk::Rect2D({0, 0}, swapChainExtent));

            vk::ClearValue clearColor(std::array<float, 4>({0.0f, 0.0f, 0.0f, 1.0f}));

            renderPassInfo.setClearValues(clearColor);

            commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

            commandBuffer.draw(3, 1, 0, 0);

            commandBuffer.endRenderPass();

            commandBuffer.end();

            i++;
        }
    }

    void createSyncObjects()
    {
        vk::SemaphoreCreateInfo semaphoreInfo;
        vk::FenceCreateInfo fenceInfo =
            vk::FenceCreateInfo()
                .setFlags(vk::FenceCreateFlagBits::eSignaled);

        imagesInFlight.resize(swapChainImages.size(), nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            imageAvailableSemaphore = device.createSemaphore(semaphoreInfo);
            renderFinishedSemaphore = device.createSemaphore(semaphoreInfo);

            inFlightFences.push_back(device.createFence(fenceInfo));
            imageAvailableSemaphores.push_back(imageAvailableSemaphore);
            renderFinishedSemaphores.push_back(renderFinishedSemaphore);
        }
    }

    /**
     * Start the engine!
     */
    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawFrame();
        }

        device.waitIdle();
    }

    void drawFrame()
    {
        vk::Fence fence = inFlightFences[currentFrame];
        device.waitForFences(fence, true, UINT64_MAX);
        uint32_t imageIndex;
        try {
            vk::ResultValue<uint32_t> res = device.acquireNextImageKHR(swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], nullptr);
        } catch(vk::OutOfDateKHRError e) {
            recreateSwapChain();
            return;
        }

        if (imagesInFlight[imageIndex])
        {
            vk::Fence fence = imagesInFlight[imageIndex];
            device.waitForFences(fence, true, UINT64_MAX);
        }

        imagesInFlight[imageIndex] = fence;

        std::vector<vk::PipelineStageFlags> waitStages = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        vk::SubmitInfo submitInfo =
            vk::SubmitInfo()
                .setWaitSemaphores(imageAvailableSemaphores[currentFrame])
                .setWaitDstStageMask(waitStages)
                .setCommandBuffers(commandBuffers[imageIndex])
                .setSignalSemaphores(renderFinishedSemaphores[currentFrame]);

        device.resetFences(fence);

        graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);

        vk::PresentInfoKHR presentInfo =
            vk::PresentInfoKHR()
                .setWaitSemaphores(renderFinishedSemaphores[currentFrame])
                .setSwapchains(swapChain)
                .setImageIndices(imageIndex);

        try {
            vk::Result result = presentQueue.presentKHR(presentInfo);
            if(result == vk::Result::eSuboptimalKHR) {
                recreateSwapChain();
            }
        } catch(vk::OutOfDateKHRError e) {
            recreateSwapChain();
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    /**
     * Clean up.
     */
    void cleanup()
    {
        cleanupSwapChain();

        for (const auto &fence : inFlightFences)
            device.destroyFence(fence);
        for (const auto &semaphore : renderFinishedSemaphores)
            device.destroySemaphore(semaphore);
        for (const auto &semaphore : imageAvailableSemaphores)
            device.destroySemaphore(semaphore);
        device.destroyCommandPool(commandPool);
        device.destroy();
        instance.destroySurfaceKHR(surface);
        instance.destroy();
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void recreateSwapChain()
    {
        device.waitIdle();

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandBuffers();
    }

    void cleanupSwapChain()
    {
        for (const auto &framebuffer : swapChainFramebuffers)
        {
            device.destroyFramebuffer(framebuffer);
        }
        swapChainFramebuffers.clear();

        device.freeCommandBuffers(commandPool, commandBuffers);
        commandBuffers.clear();

        device.destroyPipeline(graphicsPipeline);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyRenderPass(renderPass);

        for (const auto &imageView : swapChainImageViews)
        {
            device.destroyImageView(imageView);
        }
        swapChainImageViews.clear();

        device.destroySwapchainKHR(swapChain);
    }
};

int main()
{
    using namespace std;
    Playthrough playthrough;

    // On exception, write it to stderr.
    try
    {
        playthrough.Run();
    }
    catch (const std::exception &e)
    {
        cerr << e.what() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}